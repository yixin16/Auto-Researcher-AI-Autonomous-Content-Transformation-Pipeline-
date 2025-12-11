from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Any
import re
import json
from pexels_api import API
from dotenv import load_dotenv
import os
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time

load_dotenv()
logger = logging.getLogger(__name__)
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# ==========================================
# 🛠️ ROBUST JSON PARSER & HELPER CLASSES
# ==========================================

def robust_json_parse(text: str) -> Dict:
    """
    Parses JSON from LLM output, handling:
    1. Markdown code blocks (```json ... ```)
    2. C-style comments (// and /* ... */)
    3. Chatty prefixes/suffixes
    4. Trailing commas
    """
    if not text:
        return {}

    # 1. Remove C-style comments (common in some fine-tunes)
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    text = text.strip()

    # 2. Try extracting from Markdown
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        # 3. Aggressive substring extraction (Find first '{' and last '}')
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text = text[start_idx : end_idx + 1]
        else:
            # If no braces found, return empty
            return {}

    # 4. Cleanup trailing commas before closing braces
    text = re.sub(r',\s*([}\]])', r'\1', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON content: {text[:100]}...")
        return {}


class QualityScore(Enum):
    EXCELLENT = 5
    GOOD = 4
    ACCEPTABLE = 3
    NEEDS_REVISION = 2
    POOR = 1


@dataclass
class AgentOutput:
    content: Any
    quality_score: QualityScore
    revision_notes: List[str]
    confidence: float
    agent_name: str
    processing_time: float
    metadata: Dict = field(default_factory=dict)


class GenerationCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, prompt: str) -> Optional[str]:
        if prompt in self.cache:
            self.hits += 1
            return self.cache[prompt]
        self.misses += 1
        return None
    
    def set(self, prompt: str, result: str):
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[prompt] = result
    
    def get_stats(self) -> Dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hits/total:.1%}" if total > 0 else "0%"
        }


# ==========================================
# 🧠 BASE AGENT (SHARED MODEL)
# ==========================================

class BaseAgent:
    _shared_model = None
    _shared_tokenizer = None
    _model_users = 0
    
    def __init__(self, model=None, tokenizer=None, model_name="microsoft/phi-2", use_cache=True):
        self.agent_name = self.__class__.__name__
        self.use_cache = use_cache
        self.cache = GenerationCache() if use_cache else None
        
        # Share model logic to prevent OOM
        if model and tokenizer:
            self.model = model
            self.tokenizer = tokenizer
        elif BaseAgent._shared_model is not None:
            self.model = BaseAgent._shared_model
            self.tokenizer = BaseAgent._shared_tokenizer
            BaseAgent._model_users += 1
        else:
            logger.info(f"[{self.agent_name}] Loading Intelligence Engine ({model_name})...")
            
            # Auto-detect device
            if torch.cuda.is_available():
                device_map = "auto"
                quantization = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                device_map = "cpu"
                quantization = None # No quantization on CPU/MPS usually for these loaders

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization,
                device_map=device_map,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            BaseAgent._shared_model = self.model
            BaseAgent._shared_tokenizer = self.tokenizer
            BaseAgent._model_users = 1
        
        self.model.eval()
        self.metrics = {"total_generations": 0, "failed_generations": 0, "avg_generation_time": 0}

    def generate(self, prompt: str, max_new_tokens=512, temperature=0.7, top_p=0.95, stop_sequences: List[str] = None) -> str:
        cache_key = f"{prompt[:200]}_{max_new_tokens}_{temperature}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached: return cached
        
        # Format for Instruct models
        formatted_prompt = f"Instruct: {prompt}\nOutput:"
        
        try:
            start_time = time.time()
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
            
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=True, 
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            generation_time = time.time() - start_time
            output_text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            
            # Extract content after "Output:"
            if "Output:" in output_text:
                result = output_text.split("Output:")[1].strip()
            else:
                result = output_text.replace(formatted_prompt, "").strip()
            
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in result:
                        result = result.split(stop_seq)[0].strip()
            
            # Metrics
            self.metrics["total_generations"] += 1
            n = self.metrics["total_generations"]
            self.metrics["avg_generation_time"] = (self.metrics["avg_generation_time"] * (n-1) + generation_time) / n
            
            if self.cache: self.cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"[{self.agent_name}] Generation Error: {e}")
            self.metrics["failed_generations"] += 1
            return ""

    def self_evaluate(self, output: str, expected_criteria: Dict) -> Tuple[QualityScore, List[str]]:
        """Basic length/content check"""
        notes = []
        if not output or len(output.strip()) < expected_criteria.get('min_length', 10):
            notes.append("Output too short")
            return QualityScore.POOR, notes
        return QualityScore.GOOD, notes


# ==========================================
# 🕵️ CRITIC AGENT
# ==========================================

class CriticAgent(BaseAgent):
    def review_key_points(self, points: List[str], text: str) -> AgentOutput:
        start = time.time()
        # Simplified review to avoid excessive overhead
        score = QualityScore.GOOD
        notes = []
        
        if len(points) < 2:
            score = QualityScore.NEEDS_REVISION
            notes.append("Too few points extracted")
        
        # Check for vagueness
        vague_words = ['various', 'things', 'stuff', 'info']
        if any(w in " ".join(points).lower() for w in vague_words):
            score = QualityScore.ACCEPTABLE
            notes.append("Language is vague")
            
        return AgentOutput(
            content={}, 
            quality_score=score, 
            revision_notes=notes, 
            confidence=0.8, 
            agent_name=self.agent_name, 
            processing_time=time.time()-start
        )

# ==========================================
# 📝 CONTENT GENERATION AGENTS
# ==========================================

class SummarizerAgent(BaseAgent):
    async def summarize_async(self, text_chunk: str, prev_context: str = "", style: str = "executive") -> AgentOutput:
        start = time.time()
        prompt = f"""You are an executive summarizer.
CONTEXT FROM PREVIOUS SECTION: {prev_context[:300]}

TEXT TO SUMMARIZE:
{text_chunk}

INSTRUCTIONS:
1. Write a 3-4 sentence summary.
2. Focus on facts and actions.
3. Be professional. Do not use phrases like "The speaker discusses".

SUMMARY:"""
        summary = self.generate(prompt, max_new_tokens=250, temperature=0.5)
        
        # Self Correction
        sentences = [s for s in summary.split('.') if len(s.strip()) > 10]
        quality = QualityScore.GOOD
        if len(sentences) < 2: quality = QualityScore.NEEDS_REVISION
        
        return AgentOutput(
            content=summary, 
            quality_score=quality, 
            revision_notes=[],
            confidence=0.85, 
            agent_name=self.agent_name, 
            processing_time=time.time()-start
        )

class KeyPointAgent(BaseAgent):
    async def extract_points_async(self, text: str) -> AgentOutput:
        start = time.time()
        prompt = f"""Extract 4 key factual takeaways from this text.
TEXT:
{text}

FORMAT:
- Point 1
- Point 2
- Point 3
- Point 4

POINTS:"""
        output = self.generate(prompt, max_new_tokens=200, temperature=0.6)
        
        # Clean bullets
        points = []
        for line in output.splitlines():
            line = re.sub(r'^[\d\-\*\•\)]+\s*', '', line.strip())
            if len(line) > 10:
                points.append(line)
        
        return AgentOutput(
            content=points[:4], 
            quality_score=QualityScore.GOOD if len(points) >= 3 else QualityScore.ACCEPTABLE,
            revision_notes=[], 
            confidence=0.8, 
            agent_name=self.agent_name, 
            processing_time=time.time()-start
        )

class InsightAgent(BaseAgent):
    async def find_insights_async(self, text: str) -> AgentOutput:
        start = time.time()
        prompt = f"""Identify 2 deep strategic insights or implications from this text.
Avoid obvious summaries. Focus on "Why this matters".

TEXT: {text}

INSIGHTS:"""
        output = self.generate(prompt, max_new_tokens=200, temperature=0.7)
        insights = [line.strip('- *') for line in output.splitlines() if len(line) > 20][:2]
        
        return AgentOutput(
            content=insights, 
            quality_score=QualityScore.GOOD, 
            revision_notes=[], 
            confidence=0.75, 
            agent_name=self.agent_name, 
            processing_time=time.time()-start
        )

class QnAAgent(BaseAgent):
    async def generate_qna_async(self, text: str) -> AgentOutput:
        start = time.time()
        prompt = f"""Generate 2 Q&A pairs based on the text.
TEXT: {text[:1000]}

FORMAT:
Q1: Question?
A1: Answer.
Q2: Question?
A2: Answer.

Q&A:"""
        output = self.generate(prompt, max_new_tokens=300)
        
        # Regex Parsing is safer than JSON for Q&A
        qna = []
        qa_pairs = re.findall(r'Q\d*:(.*?)[\n\r]+A\d*:(.*?)(?=Q\d*:|$)', output, re.DOTALL)
        for q, a in qa_pairs:
            qna.append((q.strip(), a.strip()))
            
        return AgentOutput(
            content=qna, 
            quality_score=QualityScore.ACCEPTABLE, 
            revision_notes=[], 
            confidence=0.7, 
            agent_name=self.agent_name, 
            processing_time=time.time()-start
        )

# ==========================================
# 📊 STRUCTURED DATA AGENTS (JSON HEAVY)
# ==========================================

class ChartAgent(BaseAgent):
    def extract_data(self, text: str) -> Optional[Dict]:
        prompt = f"""Analyze if this text contains numerical data suitable for a chart (bar, line, or pie).
TEXT: {text[:1500]}

REQUIREMENTS:
1. If data exists, set found_data to true and extract values.
2. If NO data exists, set found_data to false.
3. OUTPUT ONLY JSON. DO NOT write explanations.

JSON OUTPUT:
{{
    "found_data": true,
    "chart_type": "bar",
    "title": "Chart Title",
    "labels": ["Label A", "Label B"],
    "values": [10, 20]
}}"""
        response = self.generate(prompt, max_new_tokens=250, temperature=0.1)
        data = robust_json_parse(response)
        
        # Synonym Handling for Hallucinations
        found = data.get('found_data') or data.get('foundData') or False
        
        if not found:
            return None
        
        labels = data.get('labels', [])
        values = data.get('values', [])
        
        if not labels or not values or len(labels) != len(values):
            return None
            
        return {
            "found_data": True,
            "chart_type": data.get('chart_type', 'bar'),
            "title": data.get('title', 'Data Visualization'),
            "labels": labels,
            "values": values
        }

class FactCheckerAgent(BaseAgent):
    async def verify_async(self, summary: str, source_text: str) -> AgentOutput:
        start = time.time()
        prompt = f"""Fact Check Task.
SOURCE TEXT: {source_text[:1500]}
SUMMARY TO CHECK: {summary}

INSTRUCTIONS:
1. Verify if the summary is supported by the source.
2. Flag hallucinations (facts not in source).
3. OUTPUT ONLY JSON.

JSON FORMAT:
{{
    "is_accurate": true,
    "hallucinations": []
}}"""
        response = self.generate(prompt, max_new_tokens=200, temperature=0.1)
        data = robust_json_parse(response)
        
        # Hallucination Handling: Check for common misspellings/synonyms of the key
        is_accurate = (
            data.get('is_accurate') or 
            data.get('is_correct') or 
            data.get('is_accurable') or 
            True
        )
        
        hallucinations = data.get('hallucinations', [])
        
        return AgentOutput(
            content={
                "is_accurate": is_accurate, 
                "hallucinations": hallucinations, 
                "correction_needed": not is_accurate and len(hallucinations) > 0
            },
            quality_score=QualityScore.GOOD if is_accurate else QualityScore.NEEDS_REVISION,
            revision_notes=hallucinations,
            confidence=0.9,
            agent_name=self.agent_name,
            processing_time=time.time()-start
        )

class ContextClassifier(BaseAgent):
    def classify(self, text: str) -> Dict:
        prompt = f"""Classify the following text.
TEXT: {text[:500]}

OUTPUT JSON:
{{
    "category": "Tutorial/News/Lecture/Review",
    "tone": "Formal/Casual/Urgent"
}}"""
        response = self.generate(prompt, max_new_tokens=100, temperature=0.1)
        data = robust_json_parse(response)
        return {
            "category": data.get("category", "General"),
            "tone": data.get("tone", "Neutral")
        }

# ==========================================
# 🎨 CREATIVE & UTILITY AGENTS
# ==========================================

class SWOTAgent(BaseAgent):
    async def analyze_async(self, text: str) -> AgentOutput:
        start = time.time()
        prompt = f"""Generate a SWOT Analysis from this text.
TEXT: {text[:4000]}

FORMAT:
S: [Strengths]
W: [Weaknesses]
O: [Opportunities]
T: [Threats]

SWOT:"""
        output = self.generate(prompt, max_new_tokens=400)
        
        # Regex Parsing
        swot = {}
        for key, name in [('S', 'Strengths'), ('W', 'Weaknesses'), ('O', 'Opportunities'), ('T', 'Threats')]:
            match = re.search(f"{key}:(.*?)(?=[SWOT]:|$)", output, re.DOTALL | re.IGNORECASE)
            swot[key] = match.group(1).strip() if match else f"No specific {name} identified."

        return AgentOutput(
            content=swot,
            quality_score=QualityScore.ACCEPTABLE,
            revision_notes=[],
            confidence=0.8,
            agent_name=self.agent_name,
            processing_time=time.time()-start
        )

class VisualKeywordAgent(BaseAgent):
    def get_search_term(self, text: str) -> str:
        prompt = f"""Extract ONE visual search term (concrete noun) for a stock photo representing this text.
TEXT: {text[:300]}
TERM (1-3 words):"""
        term = self.generate(prompt, max_new_tokens=15, stop_sequences=["\n"]).strip()
        # Clean punctuation
        return re.sub(r'[^\w\s]', '', term)

class TitleAgent(BaseAgent):
    def generate_title(self, text: str) -> str:
        prompt = f"""Create a short, professional title for this content.
TEXT: {text[:500]}
TITLE (Max 7 words):"""
        title = self.generate(prompt, max_new_tokens=30, stop_sequences=["\n"]).strip('" ')
        return title

class VisualAgent:
    """Wrapper for Pexels API"""
    def __init__(self):
        self.api_key = PEXELS_API_KEY
        self.api = API(self.api_key) if self.api_key else None
        self.cache = {}
        
    def get_image_for_topic(self, topic: str) -> Optional[str]:
        if not self.api: return None
        if topic in self.cache: return self.cache[topic]
        
        try:
            self.api.search(topic, page=1, results_per_page=1)
            photos = self.api.get_entries()
            if photos and hasattr(photos[0], 'original'):
                url = photos[0].original
                self.cache[topic] = url
                return url
        except Exception as e:
            logger.warning(f"Pexels Error: {e}")
        return None