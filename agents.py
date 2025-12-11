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

# --- Helper Function for JSON Parsing ---
def robust_json_parse(text: str) -> Dict:
    """
    Robustly extracts and parses JSON from a string, handling Markdown code blocks,
    conversational text after the JSON, and common LLM formatting issues.
    """
    text = text.strip()
    
    # 1. Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Extract from Markdown code blocks (```json ... ```)
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # 3. Aggressive Substring Extraction (Find first '{' and last '}')
    # This fixes the error: {"found_data": false} since there isn't any...
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = text[start_idx : end_idx + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 4. Attempt to fix common issues (trailing commas) within the substring
            try:
                # Remove trailing commas before closing braces/brackets
                json_str_fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
                return json.loads(json_str_fixed)
            except Exception:
                pass

    logger.warning(f"Failed to parse JSON content: {text[:100]}...")
    return {}

# --- Enums and Data Classes ---

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
    content: Any
    quality_score: QualityScore
    revision_notes: List[str]
    confidence: float
    agent_name: str
    processing_time: float
    metadata: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class GenerationCache:
    """Simple cache to avoid regenerating identical prompts"""
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
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))
        self.cache[prompt] = result
    
    def get_stats(self) -> Dict:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1%}",
            "cache_size": len(self.cache)
        }


class BaseAgent:
    """Enhanced base agent with caching, better error handling, and metrics"""
    
    _shared_model = None
    _shared_tokenizer = None
    _model_users = 0
    
    def __init__(self, model=None, tokenizer=None, model_name="microsoft/phi-2", use_cache=True):
        self.agent_name = self.__class__.__name__
        self.use_cache = use_cache
        self.cache = GenerationCache() if use_cache else None
        
        # Share model across agents to save memory
        if model and tokenizer:
            self.model = model
            self.tokenizer = tokenizer
        elif BaseAgent._shared_model is not None:
            self.model = BaseAgent._shared_model
            self.tokenizer = BaseAgent._shared_tokenizer
            BaseAgent._model_users += 1
            logger.info(f"[{self.agent_name}] Reusing shared model (users: {BaseAgent._model_users})")
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
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Cache the model for other agents
            BaseAgent._shared_model = self.model
            BaseAgent._shared_tokenizer = self.tokenizer
            BaseAgent._model_users = 1
        
        self.model.eval()
        self.generation_history = []
        self.metrics = {
            "total_generations": 0,
            "failed_generations": 0,
            "avg_generation_time": 0,
            "total_tokens_generated": 0
        }

    def generate(self, prompt: str, max_new_tokens=512, temperature=0.9, 
                 top_p=0.95, stop_sequences: List[str] = None) -> str:
        """Enhanced generation with caching and better error handling"""
        
        # Check cache first
        if self.cache:
            cache_key = f"{prompt[:200]}_{max_new_tokens}_{temperature}"
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug(f"[{self.agent_name}] Cache hit!")
                return cached
        
        formatted_prompt = f"Instruct: {prompt}\nOutput:"
        
        try:
            start_time = time.time()
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", 
                                   truncation=True, max_length=2048).to(self.model.device)
            
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=True, 
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                )
            
            generation_time = time.time() - start_time
            output_text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            
            # Extract actual output
            if "Output:" in output_text:
                result = output_text.split("Output:")[1].strip()
            else:
                result = output_text.replace(formatted_prompt, "").strip()
            
            # Apply stop sequences
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in result:
                        result = result.split(stop_seq)[0].strip()
            
            # Update metrics
            self.metrics["total_generations"] += 1
            self.metrics["total_tokens_generated"] += len(out[0])
            prev_avg = self.metrics["avg_generation_time"]
            n = self.metrics["total_generations"]
            self.metrics["avg_generation_time"] = (prev_avg * (n-1) + generation_time) / n
            
            # Store in history (keep last 10)
            self.generation_history.append({
                'prompt': prompt[:100],
                'output': result[:200],
                'tokens': len(out[0]),
                'time': generation_time
            })
            if len(self.generation_history) > 10:
                self.generation_history.pop(0)
            
            # Cache result
            if self.cache:
                self.cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"[{self.agent_name}] Generation Error: {e}")
            self.metrics["failed_generations"] += 1
            return ""

    def self_evaluate(self, output: str, expected_criteria: Dict) -> Tuple[QualityScore, List[str]]:
        """Enhanced self-evaluation with more sophisticated checks"""
        notes = []
        score = QualityScore.GOOD
        
        # Basic quality checks
        if not output or len(output.strip()) < expected_criteria.get('min_length', 10):
            notes.append(f"Output too short (min: {expected_criteria.get('min_length', 10)} chars)")
            score = QualityScore.POOR
            return score, notes
        
        if len(output) > expected_criteria.get('max_length', 10000):
            notes.append(f"Output exceeds max length ({expected_criteria.get('max_length', 10000)} chars)")
            score = QualityScore.NEEDS_REVISION
        
        # Check for repetition
        sentences = [s.strip() for s in output.split('.') if s.strip()]
        if len(sentences) > 3:
            unique_sentences = len(set(sentences))
            if unique_sentences < len(sentences) * 0.8:
                notes.append(f"Repetitive content detected")
                score = QualityScore.NEEDS_REVISION
        
        # Check for meaningful content
        words = output.lower().split()
        meaningful_words = [w for w in words if len(w) > 3]
        if len(meaningful_words) < len(words) * 0.4:
            notes.append("Output lacks substantive content")
            score = QualityScore.ACCEPTABLE
        
        # Check for incomplete sentences
        if output and not output.strip()[-1] in '.!?':
            notes.append("Output appears incomplete")
            if score == QualityScore.GOOD:
                score = QualityScore.ACCEPTABLE
        
        # Domain-specific checks
        if 'avoid_phrases' in expected_criteria:
            for phrase in expected_criteria['avoid_phrases']:
                if phrase.lower() in output.lower():
                    notes.append(f"Contains unwanted phrase: '{phrase}'")
                    score = QualityScore.ACCEPTABLE
        
        return score, notes
    
    def get_metrics(self) -> Dict:
        """Return agent performance metrics"""
        metrics = self.metrics.copy()
        if self.cache:
            metrics["cache_stats"] = self.cache.get_stats()
        return metrics


class CriticAgent(BaseAgent):
    """Enhanced meta-agent with multi-pass review capabilities"""
    
    def __init__(self, model=None, tokenizer=None, model_name="microsoft/phi-2"):
        super().__init__(model, tokenizer, model_name)
        self.review_history = []
        logger.info(f"[{self.agent_name}] Initialized quality control system")
    
    def review_summary(self, summary: str, original_text: str, 
                       standards: Dict = None) -> AgentOutput:
        start = time.time()
        
        standards = standards or {
            "min_accuracy": 0.8,
            "max_length_ratio": 0.3,
            "require_key_facts": True
        }
        
        prompt = f"""You are a quality reviewer. Evaluate this summary against the original text.

Original (first 600 chars): {original_text[:600]}

Summary: {summary}

Evaluation Criteria:
1. ACCURACY: Does it capture key facts without distortion?
2. CONCISENESS: Is it brief but complete?
3. CLARITY: Is it easy to understand?
4. COMPLETENESS: Are critical details included?

Provide:
- Score (1-5, where 5=excellent)
- Specific issues found
- Actionable suggestions

Respond in JSON format:
{{"score": 1-5, "issues": ["specific problems"], "suggestions": ["improvements"], "strengths": ["what works well"]}}"""
        
        response = self.generate(prompt, max_new_tokens=350, temperature=0.3)
        
        try:
            review = robust_json_parse(response)
            score_val = int(review.get('score', 3))
            quality = QualityScore(min(5, max(1, score_val)))
            issues = review.get('issues', [])
            
            self.review_history.append({
                'type': 'summary',
                'quality': quality,
                'issues': issues,
                'timestamp': datetime.now().isoformat()
            })
            
            return AgentOutput(
                content=review,
                quality_score=quality,
                revision_notes=issues,
                confidence=0.7 + (score_val / 10),
                agent_name=self.agent_name,
                processing_time=time.time() - start,
                metadata={
                    'standards_used': standards,
                    'review_type': 'summary'
                }
            )
        except Exception as e:
            logger.error(f"[{self.agent_name}] Review parsing failed: {e}")
            return AgentOutput(
                content={'score': 3, 'issues': ['Review parsing failed'], 'suggestions': []},
                quality_score=QualityScore.ACCEPTABLE,
                revision_notes=["Unable to parse review"],
                confidence=0.5,
                agent_name=self.agent_name,
                processing_time=time.time() - start
            )
    
    def aggregate_reviews(self, reviews: List[AgentOutput]) -> Dict:
        """Aggregate multiple reviews into overall assessment"""
        if not reviews:
            return {"overall_quality": "UNKNOWN", "consensus": []}
        
        scores = [r.quality_score.value for r in reviews]
        avg_score = sum(scores) / len(scores)
        
        all_issues = []
        for r in reviews:
            all_issues.extend(r.revision_notes)
        
        # Find common issues
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        consensus_issues = [issue for issue, count in issue_counts.items() 
                          if count >= len(reviews) * 0.5]
        
        return {
            "overall_quality": QualityScore(round(avg_score)).name,
            "avg_score": avg_score,
            "consensus_issues": consensus_issues,
            "all_issues": all_issues,
            "review_count": len(reviews)
        }


class SummarizerAgent(BaseAgent):
    """Enhanced summarizer with progressive refinement"""
    
    def __init__(self, model=None, tokenizer=None, model_name="microsoft/phi-2"):
        super().__init__(model, tokenizer, model_name)
        self.summary_styles = {
            "executive": "high-level overview for decision makers",
            "technical": "detailed with technical terms preserved",
            "narrative": "story-like flow with context",
            "bullet": "key points in concise bullets"
        }
    
    async def summarize_async(self, text_chunk: str, prev_context: str = "", 
                            style: str = "executive", max_retries: int = 2) -> AgentOutput:
        start = time.time()
        
        style_instruction = self.summary_styles.get(style, self.summary_styles["executive"])
        
        context_instruction = ""
        if prev_context:
            context_instruction = f"\nPREVIOUS CONTEXT: {prev_context}\nMaintain narrative continuity."

        for attempt in range(max_retries + 1):
            prompt = f"""You are an expert summarizer creating a {style} summary.

Style: {style_instruction}
{context_instruction}

TEXT TO SUMMARIZE:
{text_chunk}

REQUIREMENTS:
- 3-4 complete sentences
- Focus on concrete facts and actions
- Maintain professional tone
- Avoid vague language like "various", "several", "important"

SUMMARY:"""
            
            summary = self.generate(prompt, max_new_tokens=200, temperature=0.6,
                                  stop_sequences=["\n\n", "TEXT:", "SUMMARY:"])
            
            # Enhanced validation
            quality_score, notes = self.self_evaluate(summary, {
                'max_length': 600,
                'min_length': 80,
                'avoid_phrases': ['various', 'several', 'many different', 'in conclusion']
            })
            
            sentences = [s.strip() for s in summary.split('.') if len(s.strip()) > 10]
            if len(sentences) < 2:
                notes.append("Summary has too few sentences")
                quality_score = QualityScore.NEEDS_REVISION
            elif len(sentences) > 5:
                notes.append("Summary is too verbose")
                quality_score = QualityScore.ACCEPTABLE
            
            if not any(char.isdigit() for char in summary) and "data" in text_chunk.lower():
                notes.append("Summary may be missing numerical data")
            
            if quality_score.value >= QualityScore.ACCEPTABLE.value or attempt == max_retries:
                return AgentOutput(
                    content=summary,
                    quality_score=quality_score,
                    revision_notes=notes,
                    confidence=0.7 + (0.1 * attempt),
                    agent_name=self.agent_name,
                    processing_time=time.time() - start,
                    metadata={
                        'style': style,
                        'attempts': attempt + 1,
                        'sentence_count': len(sentences)
                    }
                )
            
            logger.warning(f"[{self.agent_name}] Attempt {attempt+1} suboptimal, revising...")
            await asyncio.sleep(0.1)
        
        return AgentOutput(
            content=summary,
            quality_score=QualityScore.ACCEPTABLE,
            revision_notes=notes,
            confidence=0.5,
            agent_name=self.agent_name,
            processing_time=time.time() - start
        )
    
    def summarize(self, text_chunk: str, style: str = "executive") -> str:
        result = asyncio.run(self.summarize_async(text_chunk, style=style))
        return result.content


class KeyPointAgent(BaseAgent):
    """Enhanced key point extraction with deduplication"""
    
    async def extract_points_async(self, text: str, target_count: int = 4,
                                  prioritize: str = "actionable") -> AgentOutput:
        start = time.time()
        
        priority_instructions = {
            "actionable": "Focus on actionable insights and decisions",
            "data": "Prioritize quantitative data and statistics",
            "concepts": "Extract key concepts and theories",
            "events": "Highlight important events and milestones"
        }
        
        priority_guide = priority_instructions.get(prioritize, priority_instructions["actionable"])
        
        prompt = f"""Extract exactly {target_count} key takeaways from this content.

PRIORITY: {priority_guide}

REQUIREMENTS for each point:
- Be specific (include numbers/names when present)
- Be concise (10-20 words)
- Be distinct (no overlap with other points)
- Be factual (no opinions or interpretations)

TEXT:
{text}

KEY TAKEAWAYS (numbered 1-{target_count}):"""
        
        output = self.generate(prompt, max_new_tokens=300, temperature=0.7,
                             stop_sequences=["\n\n", "TEXT:"])
        
        points = []
        for line in output.splitlines():
            cleaned = line.strip()
            cleaned = re.sub(r'^[\d\-\*\•\)\]]+[\.\):\s]*', '', cleaned)
            cleaned = re.sub(r'^\[.*?\]\s*', '', cleaned)
            
            if cleaned and 10 < len(cleaned) < 150 and len(cleaned.split()) >= 3:
                if not any(self._similarity(cleaned, existing) > 0.7 for existing in points):
                    points.append(cleaned)
        
        points = points[:target_count]
        
        quality_score = QualityScore.GOOD
        notes = []
        
        if len(points) < target_count * 0.75:
            notes.append(f"Only extracted {len(points)}/{target_count} points")
            quality_score = QualityScore.ACCEPTABLE
        
        vague_words = ['various', 'some', 'many', 'several', 'important', 'interesting', 'things']
        vague_count = sum(1 for p in points if any(vw in p.lower() for vw in vague_words))
        if vague_count > len(points) * 0.3:
            notes.append(f"{vague_count} points use vague language")
            quality_score = QualityScore.ACCEPTABLE
        
        specific_count = sum(1 for p in points if any(c.isdigit() for c in p) or any(w[0].isupper() for w in p.split()))
        if specific_count < len(points) * 0.5:
            notes.append("Points lack specific details")
        
        return AgentOutput(
            content=points,
            quality_score=quality_score,
            revision_notes=notes,
            confidence=0.75 + (len(points) / target_count) * 0.15,
            agent_name=self.agent_name,
            processing_time=time.time() - start,
            metadata={
                'priority': prioritize,
                'extraction_rate': f"{len(points)}/{target_count}"
            }
        )
    
    def _similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)
    
    def extract_points(self, text: str, prioritize: str = "actionable") -> List[str]:
        result = asyncio.run(self.extract_points_async(text, prioritize=prioritize))
        return result.content


class InsightAgent(BaseAgent):
    """Deep analysis with enhanced pattern recognition"""
    
    def __init__(self, model=None, tokenizer=None, model_name="microsoft/phi-2"):
        super().__init__(model, tokenizer, model_name)
        self.insight_types = {
            "trend": "emerging patterns and future directions",
            "causal": "cause-effect relationships",
            "comparative": "contrasts and comparisons",
            "implication": "consequences and impacts"
        }
    
    async def find_insights_async(self, text: str, target_count: int = 3,
                                 insight_type: str = "trend") -> AgentOutput:
        start = time.time()
        
        type_guide = self.insight_types.get(insight_type, self.insight_types["trend"])
        
        prompt = f"""You are a strategic analyst finding NON-OBVIOUS insights.

INSIGHT TYPE: {type_guide}

Real insights must:
- Reveal hidden patterns or implications
- Explain WHY something matters (not just WHAT happened)
- Predict future impact or suggest action
- Connect disparate ideas
- Go beyond surface-level observations

AVOID:
- Restating obvious facts
- Generic observations anyone could make
- Descriptions without analysis

CONTENT:
{text}

Provide {target_count} {insight_type} insights:"""
        
        output = self.generate(prompt, max_new_tokens=400, temperature=1.0,
                             stop_sequences=["\n\nCONTENT:", "INSIGHT TYPE:"])
        
        insights = []
        for line in output.splitlines():
            cleaned = re.sub(r'^[\d\-\*\•]+[\.\):\s]*', '', line.strip())
            if cleaned and len(cleaned) > 30 and len(cleaned.split()) > 5:
                if not any(meta in cleaned.lower() for meta in ['here are', 'insight:', 'analysis:', 'first,']):
                    insights.append(cleaned)
        
        insights = insights[:target_count]
        
        quality_score = QualityScore.GOOD
        notes = []
        
        analytical_words = ['because', 'therefore', 'suggests', 'implies', 'indicates', 
                          'reveals', 'demonstrates', 'leads to', 'results in', 'due to']
        analytical_count = sum(1 for ins in insights 
                             if any(aw in ins.lower() for aw in analytical_words))
        
        if analytical_count < len(insights) * 0.5:
            notes.append("Insights lack analytical depth")
            quality_score = QualityScore.ACCEPTABLE
        
        future_words = ['will', 'could', 'may', 'might', 'potential', 'likely', 'expect']
        forward_count = sum(1 for ins in insights 
                          if any(fw in ins.lower() for fw in future_words))
        
        if forward_count == 0:
            notes.append("Insights lack forward-looking perspective")
        else:
            notes.append(f"{forward_count} insights are forward-looking (good)")
        
        if len(insights) > 1:
            avg_similarity = sum(self._calculate_similarity(insights[i], insights[j])
                               for i in range(len(insights))
                               for j in range(i+1, len(insights))) / (len(insights) * (len(insights)-1) / 2)
            if avg_similarity > 0.5:
                notes.append("Insights are too similar to each other")
                quality_score = QualityScore.ACCEPTABLE
        
        confidence = 0.6 + (analytical_count / max(len(insights), 1)) * 0.3
        
        return AgentOutput(
            content=insights,
            quality_score=quality_score,
            revision_notes=notes,
            confidence=confidence,
            agent_name=self.agent_name,
            processing_time=time.time() - start,
            metadata={
                'insight_type': insight_type,
                'analytical_count': analytical_count,
                'forward_looking': forward_count
            }
        )
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)
    
    def find_insights(self, text: str, insight_type: str = "trend") -> List[str]:
        result = asyncio.run(self.find_insights_async(text, insight_type=insight_type))
        return result.content


class QnAAgent(BaseAgent):
    """Enhanced Q&A generation with Bloom's Taxonomy levels"""
    
    def __init__(self, model=None, tokenizer=None, model_name="microsoft/phi-2"):
        super().__init__(model, tokenizer, model_name)
        self.question_levels = {
            "remember": "Recall basic facts and information",
            "understand": "Explain concepts and relationships",
            "apply": "Use knowledge in new situations",
            "analyze": "Break down and examine components",
            "evaluate": "Make judgments and assessments",
            "create": "Synthesize into new ideas"
        }
    
    async def generate_qna_async(self, text: str, count: int = 2,
                                cognitive_level: str = "analyze") -> AgentOutput:
        start = time.time()
        
        level_desc = self.question_levels.get(cognitive_level, self.question_levels["analyze"])
        
        prompt = f"""Create {count} thought-provoking Q&A pairs at the {cognitive_level.upper()} level.

COGNITIVE LEVEL: {level_desc}

Questions should:
- Test deep understanding (not surface facts)
- Be open-ended when appropriate
- Challenge assumptions
- Encourage critical thinking

Answers should:
- Be comprehensive but concise (2-4 sentences)
- Reference specific details from content
- Show connections between ideas
- Demonstrate {cognitive_level}-level thinking

CONTENT:
{text}

Format:
Q1: [Question]
A1: [Answer]

Q2: [Question]
A2: [Answer]

Q&A:"""
        
        output = self.generate(prompt, max_new_tokens=450, temperature=0.8,
                             stop_sequences=["\n\nCONTENT:", "\n\nFormat:"])
        
        qna = []
        lines = output.splitlines()
        i = 0
        
        while i < len(lines) and len(qna) < count:
            line = lines[i].strip()
            if re.match(r'^Q\d*[:.\)]', line, re.IGNORECASE):
                question = re.sub(r'^Q\d*[:.\)]\s*', '', line, flags=re.IGNORECASE).strip()
                answer = ""
                for j in range(i+1, min(i+5, len(lines))):
                    ans_line = lines[j].strip()
                    if re.match(r'^A\d*[:.\)]', ans_line, re.IGNORECASE):
                        answer = re.sub(r'^A\d*[:.\)]\s*', '', ans_line, flags=re.IGNORECASE).strip()
                        k = j + 1
                        while k < len(lines) and not re.match(r'^[QA]\d*[:.\)]', lines[k].strip(), re.IGNORECASE):
                            if lines[k].strip():
                                answer += " " + lines[k].strip()
                            k += 1
                        break
                
                if question and answer and len(answer) > 20:
                    qna.append((question, answer))
                    i = k if answer else j
                else:
                    i += 1
            else:
                i += 1
        
        quality_score = QualityScore.GOOD
        notes = []
        
        if len(qna) < count * 0.75:
            notes.append(f"Generated {len(qna)}/{count} Q&A pairs")
            quality_score = QualityScore.ACCEPTABLE
        
        shallow_patterns = [r'\bwhat is\b', r'\bwho is\b', r'\bwhen did\b', r'\bwhere\b']
        shallow_count = sum(1 for q, _ in qna 
                          if any(re.search(pat, q.lower()) for pat in shallow_patterns))
        
        if shallow_count > len(qna) * 0.3 and cognitive_level not in ["remember", "understand"]:
            notes.append(f"{shallow_count} questions are too basic for {cognitive_level} level")
            quality_score = QualityScore.ACCEPTABLE
        
        short_answers = sum(1 for _, a in qna if len(a.split()) < 15)
        if short_answers > len(qna) * 0.5:
            notes.append(f"{short_answers} answers are too brief")
        
        return AgentOutput(
            content=qna,
            quality_score=quality_score,
            revision_notes=notes,
            confidence=0.7 + (len(qna) / count) * 0.2,
            agent_name=self.agent_name,
            processing_time=time.time() - start,
            metadata={
                'cognitive_level': cognitive_level,
                'shallow_questions': shallow_count
            }
        )
    
    def generate_qna(self, text: str, cognitive_level: str = "analyze") -> List[Tuple[str, str]]:
        result = asyncio.run(self.generate_qna_async(text, cognitive_level=cognitive_level))
        return result.content


class ChartAgent(BaseAgent):
    """Enhanced data extraction with validation"""
    
    def extract_data(self, text: str, chart_preference: str = "auto") -> Optional[Dict]:
        # Updated prompt to be stricter
        prompt = f"""Extract numerical data suitable for visualization.

REQUIREMENTS:
1. Look for time series, categories, or proportions.
2. Extract 3-7 data points.
3. OUTPUT ONLY JSON. NO EXPLANATION TEXT.

TEXT:
{text[:2000]}

OUTPUT STRICT JSON:
{{
    "found_data": true/false,
    "chart_type": "bar/line/pie/scatter",
    "title": "Descriptive Chart Title",
    "labels": ["Label1", "Label2"],
    "values": [10, 20],
    "unit": "unit_name",
    "context": "Brief context"
}}

JSON:"""
        
        response = self.generate(prompt, max_new_tokens=250, temperature=0.1,
                               stop_sequences=["\n\nTEXT:", "OUTPUT"])
        
        data = robust_json_parse(response)
        
        # Validation
        if not data or not data.get('found_data'):
            return None
        
        labels = data.get('labels', [])
        values = data.get('values', [])
        
        # Must have matching labels and values
        if len(labels) != len(values) or len(labels) < 2:
            return None
        
        valid_types = ['bar', 'line', 'pie', 'scatter']
        if data.get('chart_type') not in valid_types:
            data['chart_type'] = 'bar'
        
        return data


class SWOTAgent(BaseAgent):
    """Enhanced SWOT with actionable recommendations"""
    
    async def analyze_async(self, text: str, include_actions: bool = True) -> AgentOutput:
        start = time.time()
        
        action_instruction = ""
        if include_actions:
            action_instruction = "After each SWOT element, suggest one actionable step."
        
        prompt = f"""Perform a strategic SWOT Analysis from a business/organizational perspective.

{action_instruction}

CONTENT:
{text}

Provide 2-3 concise sentences for each dimension:

STRENGTHS (Internal Positive):
- What advantages exist?
- What is done well?

WEAKNESSES (Internal Negative):
- What could be improved?
- What resources are lacking?

OPPORTUNITIES (External Positive):
- What favorable conditions exist?
- What trends could be leveraged?

THREATS (External Negative):
- What obstacles exist?
- What could cause problems?

Format:
S: [analysis]
W: [analysis]
O: [analysis]
T: [analysis]

SWOT ANALYSIS:"""
        
        output = self.generate(prompt, max_new_tokens=600, temperature=0.7,
                             stop_sequences=["\n\nCONTENT:", "Format:"])
        
        swot = {"S": "", "W": "", "O": "", "T": ""}
        current_key = None
        
        for line in output.splitlines():
            line = line.strip()
            match = re.match(r'^([SWOT])[:.\)]\s*(.+)', line, re.IGNORECASE)
            if match:
                key = match.group(1).upper()
                content = match.group(2).strip()
                if key in swot:
                    current_key = key
                    swot[key] = content
            elif current_key and line and not line.startswith(('STRENGTHS', 'WEAKNESSES', 'OPPORTUNITIES', 'THREATS')):
                swot[current_key] += " " + line
        
        for key in swot:
            swot[key] = swot[key].strip()
        
        quality_score = QualityScore.GOOD
        notes = []
        
        empty_count = sum(1 for v in swot.values() if len(v.strip()) < 30)
        if empty_count > 0:
            notes.append(f"{empty_count} SWOT sections are underdeveloped")
            quality_score = QualityScore.ACCEPTABLE
        
        if include_actions:
            action_words = ['should', 'could', 'recommend', 'suggest', 'consider', 'implement']
            action_count = sum(1 for v in swot.values() 
                             if any(aw in v.lower() for aw in action_words))
            if action_count < 2:
                notes.append("Limited actionable recommendations")
        
        vague_terms = ['various', 'several', 'many', 'some', 'things']
        vague_count = sum(1 for v in swot.values() 
                        if any(vt in v.lower() for vt in vague_terms))
        if vague_count > 1:
            notes.append("Some sections lack specificity")
        
        confidence = 0.7 - (empty_count * 0.1) + (0.05 if vague_count == 0 else 0)
        
        return AgentOutput(
            content=swot,
            quality_score=quality_score,
            revision_notes=notes,
            confidence=max(0.5, min(0.95, confidence)),
            agent_name=self.agent_name,
            processing_time=time.time() - start,
            metadata={
                'empty_sections': empty_count,
                'includes_actions': include_actions
            }
        )
    
    def analyze(self, text: str, include_actions: bool = True) -> Dict:
        result = asyncio.run(self.analyze_async(text, include_actions=include_actions))
        return result.content

class FactCheckerAgent(BaseAgent):
    """Enhanced fact checker with confidence scoring"""
    
    async def verify_async(self, summary: str, source_text: str, 
                          strict_mode: bool = False) -> AgentOutput:
        start = time.time()
        
        strict_instruction = ""
        if strict_mode:
            strict_instruction = "Be VERY strict. Flag discrepancies."
        
        prompt = f"""Compare Summary vs Source.

SOURCE:
{source_text[:2000]}

SUMMARY:
{summary}

OUTPUT JSON ONLY:
{{
    "is_accurate": true/false,
    "confidence": 0.9,
    "hallucinations": ["list claims not in source"],
    "warnings": ["ambiguous points"],
    "correction_needed": true/false
}}"""
        
        response = self.generate(prompt, max_new_tokens=300, temperature=0.1)
        
        data = robust_json_parse(response)
        
        # Robust Key Extraction (Handles "is_accurable" vs "is_accurate")
        is_accurate = data.get('is_accurate', data.get('is_accurable', True))
        
        hallucinations = data.get('hallucinations', [])
        if hallucinations is None: hallucinations = [] # Handle null
        
        warnings = data.get('warnings', [])
        if warnings is None: warnings = []
        
        if is_accurate and not hallucinations and not warnings:
            score = QualityScore.EXCELLENT
        elif is_accurate and len(warnings) <= 1:
            score = QualityScore.GOOD
        elif len(hallucinations) <= 1 or not strict_mode:
            score = QualityScore.ACCEPTABLE
        else:
            score = QualityScore.NEEDS_REVISION
        
        revision_notes = hallucinations + [f"⚠️ {w}" for w in warnings]
        
        return AgentOutput(
            content=data,
            quality_score=score,
            revision_notes=revision_notes,
            confidence=float(data.get('confidence', 0.7)),
            agent_name=self.agent_name,
            processing_time=time.time() - start,
            metadata={
                'strict_mode': strict_mode,
                'hallucination_count': len(hallucinations)
            }
        )

    def verify(self, summary: str, source_text: str) -> Dict:
        """Sync wrapper"""
        result = asyncio.run(self.verify_async(summary, source_text))
        return result.content


class ContextClassifier(BaseAgent):
    """Enhanced context classification with confidence"""
    
    def classify(self, text_sample: str) -> Dict:
        prompt = f"""Analyze and classify this content.

CATEGORIES: TUTORIAL, LECTURE, NEWS, PRODUCT_REVIEW, INTERVIEW, DISCUSSION, ENTERTAINMENT, DOCUMENTARY

TONE: FORMAL, CASUAL, URGENT, INSPIRATIONAL, TECHNICAL, CONVERSATIONAL, PERSUASIVE

AUDIENCE: GENERAL, PROFESSIONAL, ACADEMIC, TECHNICAL, YOUTH

CONTENT:
{text_sample[:1200]}

OUTPUT STRICT JSON:
{{
    "category": "PRIMARY_CATEGORY",
    "secondary_category": "optional secondary",
    "tone": "PRIMARY_TONE",
    "audience": "TARGET_AUDIENCE",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

JSON:"""
        
        response = self.generate(prompt, max_new_tokens=150, temperature=0.2,
                               stop_sequences=["\n\nCONTENT:", "OUTPUT"])
        
        result = robust_json_parse(response)
        
        result.setdefault('category', 'DISCUSSION')
        result.setdefault('tone', 'CONVERSATIONAL')
        result.setdefault('audience', 'GENERAL')
        result.setdefault('confidence', 0.6)
        
        return result


class VisualKeywordAgent(BaseAgent):
    """Enhanced visual keyword generation with fallbacks"""
    
    def get_search_term(self, text: str, style: str = "professional") -> str:
        style_guides = {
            "professional": "corporate, business, professional settings",
            "abstract": "conceptual, metaphorical imagery",
            "technical": "technology, equipment, diagrams",
            "natural": "nature, outdoor, organic elements"
        }
        
        style_guide = style_guides.get(style, style_guides["professional"])
        
        prompt = f"""Generate ONE specific visual search term for stock photography.

STYLE PREFERENCE: {style_guide}

GUIDELINES:
- Use concrete nouns (not abstract concepts like 'success')
- Be specific (not 'business' but 'team meeting' or 'data analysis')
- Consider: objects, scenes, professions, activities
- Should match content theme

CONTENT:
{text[:300]}

VISUAL KEYWORD (1-4 words max):"""
        
        result = self.generate(prompt, max_new_tokens=20, temperature=0.5,
                             stop_sequences=["\n", "CONTENT:", "GUIDELINES:"]).strip()
        
        result = re.sub(r'[^\w\s-]', '', result)
        result = re.sub(r'\s+', ' ', result)
        words = result.split()[:4]
        
        if not words:
            fallbacks = {
                "professional": "business team",
                "abstract": "abstract pattern",
                "technical": "technology concept",
                "natural": "nature landscape"
            }
            return fallbacks.get(style, "abstract concept")
        
        return ' '.join(words)


class TitleAgent(BaseAgent):
    """Enhanced title generation with multiple styles"""
    
    def generate_title(self, summary_text: str, style: str = "professional") -> str:
        style_instructions = {
            "professional": "Formal, informative, suitable for business presentation",
            "engaging": "Catchy, attention-grabbing, while maintaining professionalism",
            "academic": "Scholarly, descriptive, suitable for research",
            "creative": "Unique angle, thought-provoking"
        }
        
        style_guide = style_instructions.get(style, style_instructions["professional"])
        
        prompt = f"""Create a compelling title for this content.

STYLE: {style_guide}

REQUIREMENTS:
- 4-8 words maximum
- Specific and informative
- No generic terms like "Overview" or "Summary"
- Capture main theme/value proposition

SUMMARY:
{summary_text[:400]}

TITLE:"""
        
        title = self.generate(prompt, max_new_tokens=30, temperature=0.7,
                            stop_sequences=["\n\n", "SUMMARY:", "REQUIREMENTS:"]).strip()
        
        title = re.sub(r'^["\':]+|["\':]+$', '', title)
        title = re.sub(r'\s+', ' ', title)
        title = title.strip('.')
        
        words = title.split()
        if len(words) > 9:
            title = ' '.join(words[:8]) + '...'
        elif len(words) < 2:
            title = "Content Analysis Summary"
        
        # Fixed logic for title casing
        small_words = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from', 'by', 'in', 'of'}
        words = title.split()
        title_cased = []
        for i, word in enumerate(words):
            if i == 0 or word.lower() not in small_words:
                title_cased.append(word.capitalize())
            else:
                title_cased.append(word.lower())
        
        return " ".join(title_cased)


class VisualAgent:
    """Enhanced visual search with caching and fallbacks"""
    
    def __init__(self):
        self.api_key = PEXELS_API_KEY
        self.api = API(self.api_key) if self.api_key else None
        self.cache = {}
        logger.info("[VisualAgent] Initialized" + (" with API key" if self.api else " WITHOUT API key"))
    
    def get_image_for_topic(self, topic: str, orientation: str = "landscape") -> Optional[str]:
        if not self.api:
            logger.warning("[VisualAgent] No API key configured")
            return None
        
        # Check cache
        cache_key = f"{topic}_{orientation}"
        if cache_key in self.cache:
            logger.debug(f"[VisualAgent] Cache hit for '{topic}'")
            return self.cache[cache_key]
        
        try:
            # Primary search
            self.api.search(topic, page=1, results_per_page=5)
            entries = self.api.get_entries()
            
            if entries:
                # Filter by orientation if possible
                # Note: Pexels API wrapper implementation varies; assuming standard object attributes
                for entry in entries:
                    if hasattr(entry, 'original'):
                        img_url = entry.original
                        self.cache[cache_key] = img_url
                        logger.info(f"[VisualAgent] Found image for '{topic}'")
                        return img_url
            
            # Fallback: try first word only
            first_word = topic.split()[0]
            if first_word != topic:
                logger.info(f"[VisualAgent] Trying fallback keyword: '{first_word}'")
                self.api.search(first_word, page=1, results_per_page=3)
                entries = self.api.get_entries()
                
                if entries and hasattr(entries[0], 'original'):
                    img_url = entries[0].original
                    self.cache[cache_key] = img_url
                    return img_url
            
            logger.warning(f"[VisualAgent] No images found for '{topic}'")
            return None
            
        except Exception as e:
            logger.error(f"[VisualAgent] Image fetch failed: {e}")
            return None