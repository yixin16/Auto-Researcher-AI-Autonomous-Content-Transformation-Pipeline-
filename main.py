# main.py

import asyncio
import logging
import time
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# --- 1. Robust Import Handling ---

# Agents: Try importing from agents.py or agents_enhanced.py
try:
    from agents import (
        SummarizerAgent, KeyPointAgent, InsightAgent, QnAAgent,
        ChartAgent, SWOTAgent, VisualKeywordAgent, TitleAgent,
        VisualAgent, CriticAgent, ContextClassifier, FactCheckerAgent,
        QualityScore, AgentOutput
    )
except ImportError:
    try:
        from agents import (
            SummarizerAgent, KeyPointAgent, InsightAgent, QnAAgent,
            ChartAgent, SWOTAgent, VisualKeywordAgent, TitleAgent,
            VisualAgent, CriticAgent, ContextClassifier, FactCheckerAgent,
            QualityScore, AgentOutput
        )
    except ImportError:
        raise ImportError("❌ Critical Error: Could not find 'agents.py' or 'agents_enhanced.py'.")

# RAG Engine: Handle class name mismatch
try:
    from rag_engine import EnhancedRAGEngine as RAGEngine
except ImportError:
    try:
        from rag_engine import RAGEngine
    except ImportError:
        raise ImportError("❌ Critical Error: Could not import RAGEngine from 'rag_engine.py'.")

# Utilities
try:
    from utils import download_audio, transcribe_audio_whisper, chunk_text
except ImportError:
    raise ImportError("❌ Critical Error: 'utils.py' is missing. Please ensure audio processing utilities exist.")

# Generators (Optional/Placeholder handling)
try:
    from ppt_generator import PPTGenerator
except ImportError:
    print("⚠️ Warning: 'ppt_generator.py' not found. Presentation generation will fail.")
    PPTGenerator = None

try:
    from image_generator import ImageGenerator
except ImportError:
    ImageGenerator = None  # AI Art is optional

# --- 2. Orchestrator Logic ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    """Track performance metrics"""
    total_time: float
    agent_times: Dict[str, float]
    retry_counts: Dict[str, int]
    quality_scores: Dict[str, QualityScore]
    cache_hits: int
    cache_misses: int

class ContentOrchestrator:
    """
    Enhanced Orchestrator 2.0
    Features: Rolling Context Memory, Fact-Checking, Dynamic Context Detection
    """
    
    def __init__(self, config: Dict):
        logger.info("🚀 Initializing Intelligent Content Orchestrator...")
        
        self.config = config
        # Use a smarter model for logic/reasoning
        model_name = config.get('llm_model', 'unsloth/llama-3-8b-Instruct-bnb-4bit') 
        
        # Initialize base model (shared across agents)
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch
            
            logger.info(f"Loading Intelligence Engine: {model_name}")
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
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise e
        
        # Initialize Intelligent Swarm
        logger.info("Initializing Agents...")
        self.classifier = ContextClassifier(self.model, self.tokenizer)     # Context Detection
        self.fact_checker = FactCheckerAgent(self.model, self.tokenizer)    # Hallucination Guard
        self.summarizer = SummarizerAgent(self.model, self.tokenizer)
        self.key_point = KeyPointAgent(self.model, self.tokenizer)
        self.insight = InsightAgent(self.model, self.tokenizer)
        self.qna = QnAAgent(self.model, self.tokenizer)
        self.chart = ChartAgent(self.model, self.tokenizer)
        self.swot = SWOTAgent(self.model, self.tokenizer)
        self.visual_kw = VisualKeywordAgent(self.model, self.tokenizer)
        self.title_gen = TitleAgent(self.model, self.tokenizer)
        self.visual = VisualAgent()
        self.critic = CriticAgent(self.model, self.tokenizer)
        
        # Metrics tracking
        self.metrics = ProcessingMetrics(
            total_time=0, agent_times={}, retry_counts={}, 
            quality_scores={}, cache_hits=0, cache_misses=0
        )
        
        logger.info("✅ System Operational: Memory & Fact-Checking Active")

    async def _process_parallel_tasks(self, chunk: str, summary: str) -> Dict:
        """
        Helper: Runs deep analysis tasks in parallel AFTER summary is generated.
        This balances narrative flow (serial) with performance (parallel).
        """
        tasks = [
            self.key_point.extract_points_async(chunk),
            self.insight.find_insights_async(chunk),
            self.qna.generate_qna_async(chunk)
        ]
        
        results = await asyncio.gather(*tasks)
        points_out, insights_out, qna_out = results
        
        # Run Critic on key points
        points_review = self.critic.review_key_points(points_out.content, chunk) if hasattr(self.critic, 'review_key_points') else None
        
        # Chart and Visuals (Synchronous/Fast)
        chart_data = self.chart.extract_data(chunk)
        visual_kw = self.visual_kw.get_search_term(summary)
        
        # Safe quality score extraction
        points_score = points_review.quality_score.name if points_review else "UNKNOWN"
        insights_score = insights_out.quality_score.name if hasattr(insights_out, 'quality_score') else "UNKNOWN"

        return {
            'points': points_out.content,
            'insights': insights_out.content,
            'qna': qna_out.content,
            'chart_data': chart_data,
            'image_prompt': visual_kw,
            'quality_data': {
                'points_score': points_score,
                'insights_score': insights_score
            }
        }

    async def step_1_analyze_async(self, url: str) -> Tuple[List[Dict], RAGEngine]:
        """
        Intelligent Analysis Pipeline:
        1. Context Detection (Tone/Category)
        2. Serial Summarization with Rolling Memory
        3. Fact Checking Loop
        4. Parallel Deep Analysis
        """
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("STEP 1: INTELLIGENT DEEP ANALYSIS")
        logger.info("=" * 60)
        
        # 1. Ingest
        logger.info("📥 Ingesting media...")
        audio_path = download_audio(url)
        # Using larger model size for better accuracy if config allows
        transcript = transcribe_audio_whisper(audio_path, self.config.get('transcription_model_size', 'base'))
        
        if not transcript:
            raise ValueError("Empty transcript produced")
            
        # 2. Context Classification
        logger.info("🧠 Analyzing content DNA (Tone & Category)...")
        # Sample middle of text to avoid intro/outro noise
        mid_point = len(transcript) // 2
        sample_text = transcript[max(0, mid_point-500) : min(len(transcript), mid_point+500)]
        context_data = self.classifier.classify(sample_text)
        logger.info(f"   Detected: {context_data.get('category', 'General')} | Tone: {context_data.get('tone', 'Neutral')}")
        
        # 3. Chunking
        chunks = chunk_text(transcript, max_len=2500)
        logger.info(f"📚 Processing {len(chunks)} chunks with Rolling Memory...")
        
        outline = []
        previous_summary_context = ""
        
        # 4. Processing Loop
        for i, chunk in enumerate(chunks):
            # Extract text from chunk dict if using utils.chunk_text
            chunk_text_content = chunk['text'] if isinstance(chunk, dict) else chunk
            
            logger.info(f"[Chunk {i+1}/{len(chunks)}] Processing...")
            
            # A. Generate Summary with Memory (Sequential)
            summary_out = await self.summarizer.summarize_async(
                chunk_text_content, 
                prev_context=previous_summary_context 
            )
            
            # B. Fact Check (Hallucination Guard)
            verification = await self.fact_checker.verify_async(summary_out.content, chunk_text_content)
            current_summary = summary_out.content
            
            # Self-Correction Loop
            if verification.content.get('correction_needed', False):
                logger.warning(f"   ⚠ Fact Check Failed: {verification.content.get('hallucinations')}")
                logger.info("   🔄 Regenerating with strict factual constraints...")
                
                correction_prompt = f"""
                PREVIOUS ERROR: Your last summary contained these hallucinations: {verification.content.get('hallucinations')}.
                REVISED INSTRUCTION: Summarize strictly based on the text provided below. Do not invent numbers.
                Original Text: {chunk_text_content}
                """
                # Retry
                summary_out = await self.summarizer.summarize_async(correction_prompt)
                current_summary = summary_out.content
                self.metrics.retry_counts['fact_check'] = self.metrics.retry_counts.get('fact_check', 0) + 1

            # C. Update Rolling Context (Keep last 400 chars for next iteration)
            previous_summary_context = current_summary[-400:]
            
            # D. Parallel Deep Analysis
            analysis_data = await self._process_parallel_tasks(chunk_text_content, current_summary)
            
            # Compile Data
            chunk_data = {
                'id': i,
                'original_text': chunk_text_content,
                'summary': current_summary,
                'points': analysis_data['points'],
                'insights': analysis_data['insights'],
                'qna': analysis_data['qna'],
                'chart_data': analysis_data['chart_data'],
                'image_prompt': analysis_data['image_prompt'],
                'quality_report': analysis_data['quality_data'],
                'meta': {
                    'tone_detected': context_data.get('tone'),
                    'quality_score': summary_out.quality_score.name
                }
            }
            
            outline.append(chunk_data)
            
            # Track Metrics
            self.metrics.quality_scores[f"summary_{i}"] = summary_out.quality_score
            self.metrics.agent_times['summarizer'] = \
                self.metrics.agent_times.get('summarizer', 0) + summary_out.processing_time

        # 5. RAG Indexing
        logger.info("📇 Building Semantic Knowledge Base...")
        rag = RAGEngine()
        # chunk_text returns list of dicts, which rag expects
        rag.ingest_transcript(chunks)
        
        self.metrics.total_time = time.time() - start_time
        logger.info(f"✅ Analysis Complete in {self.metrics.total_time:.1f}s")
        logger.info(f"📊 Quality Summary: {self._summarize_quality()}")
        
        return outline, rag

    def step_1_analyze(self, url: str) -> Tuple[List[Dict], RAGEngine]:
        """Sync wrapper"""
        return asyncio.run(self.step_1_analyze_async(url))
    
    def _summarize_quality(self) -> str:
        """Generate quality summary string"""
        if not self.metrics.quality_scores:
            return "N/A"
        
        scores = list(self.metrics.quality_scores.values())
        avg_score = sum(s.value for s in scores) / len(scores)
        
        excellent = sum(1 for s in scores if s == QualityScore.EXCELLENT)
        good = sum(1 for s in scores if s == QualityScore.GOOD)
        acceptable = sum(1 for s in scores if s == QualityScore.ACCEPTABLE)
        needs_work = sum(1 for s in scores if s.value < QualityScore.ACCEPTABLE.value)
        
        return f"Avg={avg_score:.1f} (⭐{excellent} ✓{good} ○{acceptable} ⚠{needs_work})"

    async def step_2_generate_async(
        self, 
        outline: List[Dict], 
        url: str,
        use_ai_images: bool = False
    ) -> Tuple[str, str, Dict]:
        """
        Enhanced presentation generation with dynamic layouts
        """
        if not PPTGenerator:
            raise ImportError("PPTGenerator not found. Cannot generate presentation.")

        logger.info("=" * 60)
        logger.info("STEP 2: INTELLIGENT PRESENTATION GENERATION")
        logger.info("=" * 60)
        
        ppt = PPTGenerator()
        img_gen = ImageGenerator() if (use_ai_images and ImageGenerator) else None
        
        # === TITLE GENERATION ===
        logger.info("📝 Generating title...")
        full_summary = " ".join(s.get('summary', '') for s in outline[:3])
        title = self.title_gen.generate_title(full_summary)
        logger.info(f"✅ Title: {title}")
        
        # === TITLE SLIDE ===
        ppt.add_title_slide(title, f"AI-Generated Analysis | {len(outline)} Sections")
        
        # === PROCESS EACH SECTION ===
        for section in outline:
            sec_id = section.get('id', 0)
            summary = section.get('summary', '')
            points = section.get('points', [])
            insights = section.get('insights', [])
            image_prompt = section.get('image_prompt', '')
            chart_data = section.get('chart_data') 
            qna_data = section.get('qna', [])

            logger.info(f"[Section {sec_id+1}] Building slides...")
            
            # Slide 1: Visual Overview
            img_url = None
            if use_ai_images and img_gen:
                logger.info("  🎨 Generating AI artwork...")
                img_url = img_gen.generate(image_prompt, f"sec{sec_id}")
            else:
                logger.info("  📷 Fetching stock photo...")
                img_url = self.visual.get_image_for_topic(image_prompt)
            
            ppt.add_visual_slide(
                f"Section {sec_id+1}: Overview",
                summary,
                img_url
            )
            
            # Slide 2: Analysis
            if len(points) <= 4 and len(insights) <= 3:
                ppt.add_analysis_slide(f"Analysis", points, insights)
            else:
                ppt.add_bullet_slide(f"Key Findings", points, icon="▸")
                if insights:
                    ppt.add_bullet_slide(f"Strategic Insights", insights, icon="💡")
            
            # Slide 3: Chart
            if chart_data:
                logger.info("  📊 Creating data visualization...")
                chart_path = ppt.create_chart_image(chart_data, f"sec{sec_id}")
                if chart_path:
                    ppt.add_chart_slide(chart_data.get('title', 'Data Analysis'), chart_path)
            
            # Slide 4: Q&A
            if qna_data:
                ppt.add_qna_slide(qna_data)
        
        # === FINAL SWOT SLIDE ===
        logger.info("📈 Performing executive SWOT analysis...")
        full_text = " ".join(s.get('original_text', '') for s in outline)
        swot_result = await self.swot.analyze_async(full_text[:5000])
        swot_data = swot_result.content
        
        ppt.add_swot_slide(swot_data)
        
        # === SAVE ===
        logger.info("💾 Saving presentation...")
        path = ppt.save(title)
        
        if img_gen:
            img_gen.unload_model()
        
        logger.info(f"✅ PRESENTATION GENERATED: {path}")
        
        return str(path), title, swot_data
    
    def step_2_generate(
        self, 
        outline: List[Dict], 
        url: str,
        use_ai_images: bool = False
    ) -> Tuple[str, str, Dict]:
        """Sync wrapper"""
        return asyncio.run(
            self.step_2_generate_async(outline, url, use_ai_images)
        )
    
    def get_performance_report(self) -> Dict:
        """Generate detailed performance metrics"""
        return {
            'total_time': self.metrics.total_time,
            'agent_times': self.metrics.agent_times,
            'retry_counts': self.metrics.retry_counts,
            'quality_summary': self._summarize_quality(),
            'quality_scores': self.metrics.quality_scores,
            'cache_stats': {
                'hits': self.metrics.cache_hits,
                'misses': self.metrics.cache_misses,
                'hit_rate': f"{(self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses) * 100):.1f}%" 
                    if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else "0%"
            }
        }