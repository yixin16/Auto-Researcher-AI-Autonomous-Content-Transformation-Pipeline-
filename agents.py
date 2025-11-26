from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import logging
import asyncio
from typing import List, Dict, Optional, Tuple
import re
import json
from pexels_api import API
from dotenv import load_dotenv
import os
from dataclasses import dataclass
from enum import Enum

load_dotenv()
logger = logging.getLogger(__name__)
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")


class QualityScore(Enum):
    """Quality assessment levels"""
    EXCELLENT = 5
    GOOD = 4
    ACCEPTABLE = 3
    NEEDS_REVISION = 2
    POOR = 1


@dataclass
class AgentOutput:
    """Structured output from agents with quality metadata"""
    content: any
    quality_score: QualityScore
    revision_notes: List[str]
    confidence: float
    agent_name: str
    processing_time: float


class BaseAgent:
    """Enhanced base agent with self-monitoring capabilities"""
    
    def __init__(self, model=None, tokenizer=None, model_name="microsoft/phi-2"):
        self.agent_name = self.__class__.__name__
        
        if model and tokenizer:
            self.model = model
            self.tokenizer = tokenizer
        else:
            logger.info(f"[{self.agent_name}] Loading Intelligence Engine ({model_name})...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        self.model.eval()
        self.generation_history = []

    def generate(self, prompt, max_new_tokens=512, temperature=0.9, top_p=0.95):
        """Enhanced generation with better control"""
        formatted_prompt = f"Instruct: {prompt}\nOutput:"
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=True, 
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Reduce repetition
                )
            output_text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            
            # Extract actual output
            if "Output:" in output_text:
                result = output_text.split("Output:")[1].strip()
            else:
                result = output_text.replace(formatted_prompt, "").strip()
            
            # Store generation history for analysis
            self.generation_history.append({
                'prompt': prompt[:100],
                'output': result[:200],
                'tokens': len(out[0])
            })
            
            return result
        except Exception as e:
            logger.error(f"[{self.agent_name}] Generation Error: {e}")
            return ""

    def self_evaluate(self, output: str, expected_criteria: Dict) -> Tuple[QualityScore, List[str]]:
        """Self-evaluation mechanism - agents check their own output quality"""
        notes = []
        score = QualityScore.GOOD
        
        # Basic quality checks
        if not output or len(output.strip()) < 10:
            notes.append("Output too short or empty")
            score = QualityScore.POOR
        
        if len(output) > expected_criteria.get('max_length', 10000):
            notes.append("Output exceeds expected length")
            score = QualityScore.NEEDS_REVISION
        
        # Check for repetition
        words = output.lower().split()
        if len(words) != len(set(words)) and len(words) > 50:
            repetition_rate = 1 - (len(set(words)) / len(words))
            if repetition_rate > 0.3:
                notes.append(f"High repetition detected ({repetition_rate:.1%})")
                score = QualityScore.NEEDS_REVISION
        
        # Check for meaningful content (not just filler words)
        meaningful_words = [w for w in words if len(w) > 3]
        if len(meaningful_words) < len(words) * 0.5:
            notes.append("Output lacks substantive content")
            score = QualityScore.ACCEPTABLE
        
        return score, notes


class CriticAgent(BaseAgent):
    """Meta-agent that reviews and critiques other agents' outputs"""
    
    def __init__(self, model=None, tokenizer=None, model_name="microsoft/phi-2"):
        super().__init__(model, tokenizer, model_name)
        self.review_history = []
        logger.info(f"[{self.agent_name}] Initialized quality control system")
    
    def review_summary(self, summary: str, original_text: str) -> AgentOutput:
        """Reviews summary quality against source material"""
        import time
        start = time.time()
        
        prompt = f"""You are a quality reviewer. Evaluate this summary against the original text.

Original (first 500 chars): {original_text[:500]}

Summary: {summary}

Rate the summary on:
1. Accuracy - Does it capture key facts?
2. Conciseness - Is it appropriately brief?
3. Clarity - Is it easy to understand?
4. Completeness - Does it miss critical information?

Respond in JSON format:
{{"score": 1-5, "issues": ["list", "of", "problems"], "suggestions": ["improvements"]}}"""
        
        response = self.generate(prompt, max_new_tokens=300, temperature=0.3)
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                review = json.loads(json_match.group())
                score_val = int(review.get('score', 3))
                quality = QualityScore(min(5, max(1, score_val)))
                issues = review.get('issues', [])
                
                self.review_history.append({
                    'type': 'summary',
                    'quality': quality,
                    'issues': issues
                })
                
                return AgentOutput(
                    content=review,
                    quality_score=quality,
                    revision_notes=issues,
                    confidence=0.7 + (score_val / 10),
                    agent_name=self.agent_name,
                    processing_time=time.time() - start
                )
        except:
            pass
        
        # Fallback if JSON parsing fails
        return AgentOutput(
            content={'score': 3, 'issues': [], 'suggestions': []},
            quality_score=QualityScore.ACCEPTABLE,
            revision_notes=["Unable to parse review"],
            confidence=0.5,
            agent_name=self.agent_name,
            processing_time=time.time() - start
        )
    
    def review_key_points(self, points: List[str], context: str) -> AgentOutput:
        """Reviews extracted key points for relevance and accuracy"""
        import time
        start = time.time()
        
        prompt = f"""Evaluate these key points extracted from content. Are they truly KEY points?

Context (first 300 chars): {context[:300]}

Points:
{chr(10).join(f'{i+1}. {p}' for i, p in enumerate(points))}

For each point, assess:
- Relevance (Is it actually important?)
- Specificity (Is it concrete, not vague?)
- Clarity (Is it understandable?)

Respond: KEEP, REVISE, or REMOVE for each point with brief reason.
Format: "1: KEEP - specific data point" or "2: REVISE - too vague, suggest: ..."
"""
        
        response = self.generate(prompt, max_new_tokens=350, temperature=0.4)
        
        # Parse feedback
        revision_notes = []
        keep_count = response.upper().count('KEEP')
        revise_count = response.upper().count('REVISE')
        remove_count = response.upper().count('REMOVE')
        
        # Quality score based on how many points need work
        if revise_count + remove_count == 0:
            quality = QualityScore.EXCELLENT
        elif revise_count + remove_count <= len(points) * 0.3:
            quality = QualityScore.GOOD
        elif revise_count + remove_count <= len(points) * 0.5:
            quality = QualityScore.ACCEPTABLE
        else:
            quality = QualityScore.NEEDS_REVISION
            revision_notes.append(f"{revise_count + remove_count} points need improvement")
        
        return AgentOutput(
            content=response,
            quality_score=quality,
            revision_notes=revision_notes,
            confidence=0.6 + (keep_count / len(points)) * 0.3,
            agent_name=self.agent_name,
            processing_time=time.time() - start
        )
    
    def review_insights(self, insights: List[str], context: str) -> AgentOutput:
        """Reviews insights for depth and non-obviousness"""
        import time
        start = time.time()
        
        prompt = f"""Evaluate these 'insights'. Real insights are NON-OBVIOUS, forward-looking, or reveal hidden patterns.

Context: {context[:300]}

Insights:
{chr(10).join(f'{i+1}. {ins}' for i, ins in enumerate(insights))}

For each, classify as:
- DEEP: Non-obvious, valuable perspective
- SURFACE: Obvious observation that anyone could make
- SPECULATIVE: Interesting but needs grounding

Format: "1: DEEP - explains why..." or "2: SURFACE - just restates facts"
"""
        
        response = self.generate(prompt, max_new_tokens=400, temperature=0.4)
        
        deep_count = response.upper().count('DEEP')
        surface_count = response.upper().count('SURFACE')
        
        # Quality based on insight depth
        if deep_count >= len(insights) * 0.7:
            quality = QualityScore.EXCELLENT
        elif deep_count >= len(insights) * 0.5:
            quality = QualityScore.GOOD
        elif surface_count > deep_count:
            quality = QualityScore.NEEDS_REVISION
        else:
            quality = QualityScore.ACCEPTABLE
        
        notes = []
        if surface_count > len(insights) * 0.4:
            notes.append(f"{surface_count} insights are too surface-level")
        
        return AgentOutput(
            content=response,
            quality_score=quality,
            revision_notes=notes,
            confidence=0.5 + (deep_count / len(insights)) * 0.4,
            agent_name=self.agent_name,
            processing_time=time.time() - start
        )


class SummarizerAgent(BaseAgent):
    """Enhanced summarizer with self-correction"""
    
    async def summarize_async(self, text_chunk: str, max_retries: int = 2) -> AgentOutput:
        """Async summarization with automatic retry on poor quality"""
        import time
        start = time.time()
        
        for attempt in range(max_retries + 1):
            prompt = f"""Summarize this transcript section into a concise executive summary (max 3 sentences).
Focus on: core message, key data/facts, and strategic implications.

Text: {text_chunk}

Executive Summary:"""
            
            summary = self.generate(prompt, max_new_tokens=150, temperature=0.6)
            
            # Self-evaluation
            quality_score, notes = self.self_evaluate(summary, {
                'max_length': 500,
                'min_length': 50
            })
            
            # Check sentence count
            sentences = [s.strip() for s in summary.split('.') if s.strip()]
            if len(sentences) > 4:
                notes.append("Summary exceeds 3 sentences")
                quality_score = QualityScore.NEEDS_REVISION
            
            if quality_score.value >= QualityScore.ACCEPTABLE.value or attempt == max_retries:
                return AgentOutput(
                    content=summary,
                    quality_score=quality_score,
                    revision_notes=notes,
                    confidence=0.7 if attempt == 0 else 0.8,
                    agent_name=self.agent_name,
                    processing_time=time.time() - start
                )
            
            logger.warning(f"[{self.agent_name}] Attempt {attempt+1} produced low quality, retrying...")
            await asyncio.sleep(0.1)
        
        # Should never reach here
        return AgentOutput(
            content=summary,
            quality_score=QualityScore.ACCEPTABLE,
            revision_notes=notes,
            confidence=0.5,
            agent_name=self.agent_name,
            processing_time=time.time() - start
        )
    
    def summarize(self, text_chunk: str) -> str:
        """Sync wrapper for backward compatibility"""
        result = asyncio.run(self.summarize_async(text_chunk))
        return result.content


class KeyPointAgent(BaseAgent):
    """Enhanced key point extraction with validation"""
    
    async def extract_points_async(self, text: str, target_count: int = 4) -> AgentOutput:
        import time
        start = time.time()
        
        prompt = f"""Extract exactly {target_count} factual key takeaways. Each should be:
- Specific (include numbers/data if present)
- Concise (max 15 words)
- Actionable or informative

Text: {text}

Key Takeaways:"""
        
        output = self.generate(prompt, max_new_tokens=250, temperature=0.7)
        
        # Parse points
        points = []
        for line in output.splitlines():
            cleaned = line.strip()
            # Remove numbering/bullets
            cleaned = re.sub(r'^[\d\-\*\•]+[\.\):\s]*', '', cleaned)
            if cleaned and len(cleaned) > 10 and len(cleaned.split()) >= 3:
                points.append(cleaned)
        
        points = points[:target_count]
        
        # Validation
        quality_score = QualityScore.GOOD
        notes = []
        
        if len(points) < target_count:
            notes.append(f"Only extracted {len(points)}/{target_count} points")
            quality_score = QualityScore.ACCEPTABLE
        
        # Check for vague points
        vague_words = ['various', 'some', 'many', 'several', 'important', 'interesting']
        vague_count = sum(1 for p in points if any(vw in p.lower() for vw in vague_words))
        if vague_count > len(points) * 0.3:
            notes.append(f"{vague_count} points contain vague language")
            quality_score = QualityScore.ACCEPTABLE
        
        return AgentOutput(
            content=points,
            quality_score=quality_score,
            revision_notes=notes,
            confidence=0.75,
            agent_name=self.agent_name,
            processing_time=time.time() - start
        )
    
    def extract_points(self, text: str) -> List[str]:
        """Sync wrapper"""
        result = asyncio.run(self.extract_points_async(text))
        return result.content


class InsightAgent(BaseAgent):
    """Deep analysis agent with enhanced prompting"""
    
    async def find_insights_async(self, text: str, target_count: int = 3) -> AgentOutput:
        import time
        start = time.time()
        
        prompt = f"""You are a strategic analyst. Provide {target_count} NON-OBVIOUS insights about this content.

Real insights should:
- Reveal hidden patterns or implications
- Explain WHY something matters (not just WHAT)
- Predict future impact or consequences
- Connect disparate ideas

Avoid: Restating obvious facts, generic observations

Text: {text}

Strategic Insights:"""
        
        output = self.generate(prompt, max_new_tokens=350, temperature=1.1)
        
        insights = []
        for line in output.splitlines():
            cleaned = re.sub(r'^[\d\-\*\•]+[\.\):\s]*', '', line.strip())
            if cleaned and len(cleaned) > 20:
                insights.append(cleaned)
        
        insights = insights[:target_count]
        
        # Quality assessment
        quality_score = QualityScore.GOOD
        notes = []
        
        # Check for forward-looking language
        future_indicators = ['will', 'could', 'may', 'suggests', 'implies', 'indicates', 'potential']
        forward_count = sum(1 for ins in insights if any(fi in ins.lower() for fi in future_indicators))
        
        if forward_count == 0:
            notes.append("Insights lack forward-looking perspective")
            quality_score = QualityScore.ACCEPTABLE
        
        return AgentOutput(
            content=insights,
            quality_score=quality_score,
            revision_notes=notes,
            confidence=0.65,
            agent_name=self.agent_name,
            processing_time=time.time() - start
        )
    
    def find_insights(self, text: str) -> List[str]:
        """Sync wrapper"""
        result = asyncio.run(self.find_insights_async(text))
        return result.content


class QnAAgent(BaseAgent):
    """Enhanced Q&A generation"""
    
    async def generate_qna_async(self, text: str, count: int = 2) -> AgentOutput:
        import time
        start = time.time()
        
        prompt = f"""Create {count} insightful Q&A pairs that test understanding of this content.

Questions should:
- Probe deeper meaning (not surface facts)
- Be open-ended when appropriate
- Challenge assumptions

Answers should:
- Be comprehensive but concise
- Reference specific details
- Show connections between ideas

Text: {text}

Q&A:"""
        
        output = self.generate(prompt, max_new_tokens=350, temperature=0.8)
        
        qna = []
        current_q = ""
        
        for line in output.splitlines():
            line = line.strip()
            if line.startswith(('Q:', 'Question:')):
                current_q = re.sub(r'^(Q:|Question:)\s*', '', line).strip()
            elif line.startswith(('A:', 'Answer:')) and current_q:
                answer = re.sub(r'^(A:|Answer:)\s*', '', line).strip()
                if answer:
                    qna.append((current_q, answer))
                    current_q = ""
        
        qna = qna[:count]
        
        # Validation
        quality_score = QualityScore.GOOD
        notes = []
        
        if len(qna) < count:
            notes.append(f"Generated {len(qna)}/{count} Q&A pairs")
            quality_score = QualityScore.ACCEPTABLE
        
        # Check question depth
        shallow_words = ['what is', 'who is', 'when did', 'where is']
        shallow_count = sum(1 for q, _ in qna if any(sw in q.lower() for sw in shallow_words))
        if shallow_count > len(qna) * 0.5:
            notes.append(f"{shallow_count} questions are too basic")
            quality_score = QualityScore.ACCEPTABLE
        
        return AgentOutput(
            content=qna,
            quality_score=quality_score,
            revision_notes=notes,
            confidence=0.7,
            agent_name=self.agent_name,
            processing_time=time.time() - start
        )
    
    def generate_qna(self, text: str) -> List[Tuple[str, str]]:
        """Sync wrapper"""
        result = asyncio.run(self.generate_qna_async(text))
        return result.content


class ChartAgent(BaseAgent):
    """Smart chart data extraction"""
    
    def extract_data(self, text: str) -> Optional[Dict]:
        prompt = f"""Analyze this text for quantitative data suitable for visualization.

Text: {text}

If you find comparable numbers (e.g., "Sales: Q1=$50M, Q2=$65M"), output JSON:
{{"title": "Quarterly Sales", "labels": ["Q1","Q2"], "values": [50, 65], "unit": "M$"}}

If no clear numerical comparison exists, output: NO_DATA

Output:"""
        
        response = self.generate(prompt, max_new_tokens=200, temperature=0.2)
        
        if "NO_DATA" in response.upper():
            return None
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                # Validate structure
                if all(k in data for k in ['title', 'labels', 'values']):
                    if len(data['labels']) == len(data['values']):
                        return data
        except:
            pass
        
        return None


class SWOTAgent(BaseAgent):
    """Strategic SWOT analysis"""
    
    async def analyze_async(self, text: str) -> AgentOutput:
        import time
        start = time.time()
        
        prompt = f"""Perform a strategic SWOT Analysis on this content from a business/organizational perspective.

Text: {text}

Provide 2-3 sentence analysis for each:
- Strengths: Internal positive attributes
- Weaknesses: Internal limitations
- Opportunities: External favorable conditions
- Threats: External risks

Format:
S: [analysis]
W: [analysis]
O: [analysis]
T: [analysis]

SWOT:"""
        
        output = self.generate(prompt, max_new_tokens=500, temperature=0.7)
        
        swot = {"S": "", "W": "", "O": "", "T": ""}
        current_key = None
        
        for line in output.splitlines():
            line = line.strip()
            if line.startswith('S:'):
                current_key = 'S'
                swot['S'] = line[2:].strip()
            elif line.startswith('W:'):
                current_key = 'W'
                swot['W'] = line[2:].strip()
            elif line.startswith('O:'):
                current_key = 'O'
                swot['O'] = line[2:].strip()
            elif line.startswith('T:'):
                current_key = 'T'
                swot['T'] = line[2:].strip()
            elif current_key and line:
                swot[current_key] += " " + line
        
        # Validation
        quality_score = QualityScore.GOOD
        notes = []
        
        empty_count = sum(1 for v in swot.values() if len(v.strip()) < 20)
        if empty_count > 0:
            notes.append(f"{empty_count} SWOT sections are underdeveloped")
            quality_score = QualityScore.ACCEPTABLE
        
        return AgentOutput(
            content=swot,
            quality_score=quality_score,
            revision_notes=notes,
            confidence=0.7,
            agent_name=self.agent_name,
            processing_time=time.time() - start
        )
    
    def analyze(self, text: str) -> Dict:
        """Sync wrapper"""
        result = asyncio.run(self.analyze_async(text))
        return result.content


class VisualKeywordAgent(BaseAgent):
    """Semantic visual keyword extraction"""
    
    def get_search_term(self, text: str) -> str:
        prompt = f"""Generate ONE specific visual search term for stock photo/image that represents this content.

Guidelines:
- Use concrete nouns (not abstract concepts)
- Be specific (not 'business' but 'financial analyst')
- Consider: objects, scenes, professions, activities

Text: {text[:200]}

Visual keyword (1-3 words):"""
        
        result = self.generate(prompt, max_new_tokens=15, temperature=0.4).strip()
        # Clean up
        result = re.sub(r'[^\w\s]', '', result)
        words = result.split()[:3]
        return ' '.join(words) if words else 'abstract concept'


class TitleAgent(BaseAgent):
    """Professional title generation"""
    
    def generate_title(self, summary_text: str) -> str:
        prompt = f"""Create a professional presentation title (4-7 words max). Make it:
- Specific and informative
- Professional tone
- Action-oriented when possible

Summary: {summary_text[:300]}

Title:"""
        
        title = self.generate(prompt, max_new_tokens=25, temperature=0.6).strip()
        # Clean
        title = re.sub(r'^["\']|["\']$', '', title)
        title = re.sub(r'\s+', ' ', title)
        # Limit length
        words = title.split()[:8]
        return ' '.join(words)


class VisualAgent:
    """Stock photo integration"""
    
    def __init__(self):
        self.api_key = PEXELS_API_KEY
        self.api = API(self.api_key) if self.api_key else None
    
    def get_image_for_topic(self, topic: str) -> Optional[str]:
        if not self.api:
            return None
        
        try:
            # Try primary keyword first
            self.api.search(topic, page=1, results_per_page=3)
            entries = self.api.get_entries()
            
            if entries:
                # Return highest quality image
                return entries[0].original
            
            # Fallback: try first word only
            first_word = topic.split()[0]
            self.api.search(first_word, page=1, results_per_page=1)
            entries = self.api.get_entries()
            
            if entries:
                return entries[0].original
        except Exception as e:
            logger.error(f"[VisualAgent] Image fetch failed: {e}")
        
        return None