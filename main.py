import asyncio
import logging
from typing import List, Dict, Tuple, Optional
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from agents import (
    SummarizerAgent, KeyPointAgent, InsightAgent, QnAAgent,
    ChartAgent, SWOTAgent, VisualKeywordAgent, TitleAgent,
    VisualAgent, CriticAgent, AgentOutput, QualityScore
)
from utils import download_audio, transcribe_audio_whisper, chunk_text
from rag_engine import RAGEngine
from ppt_generator import PPTGenerator
from image_generator import ImageGenerator
from cache_manager import get_or_create
import hashlib

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
    """Enhanced orchestrator with parallel processing and quality control"""
    
    def __init__(self, config: Dict):
        logger.info("🚀 Initializing Enhanced Content Orchestrator...")
        
        self.config = config
        model_name = config.get('llm_model', 'microsoft/phi-2')
        
        # Initialize base model (shared across agents)
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        
        logger.info(f"Loading shared model: {model_name}")
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
        
        # Initialize agents with shared model
        logger.info("Initializing intelligent agents...")
        self.summarizer = SummarizerAgent(self.model, self.tokenizer)
        self.key_point = KeyPointAgent(self.model, self.tokenizer)
        self.insight = InsightAgent(self.model, self.tokenizer)
        self.qna = QnAAgent(self.model, self.tokenizer)
        self.chart = ChartAgent(self.model, self.tokenizer)
        self.swot = SWOTAgent(self.model, self.tokenizer)
        self.visual_kw = VisualKeywordAgent(self.model, self.tokenizer)
        self.title_gen = TitleAgent(self.model, self.tokenizer)
        self.visual = VisualAgent()
        
        # Initialize critic agent (quality control)
        self.critic = CriticAgent(self.model, self.tokenizer)
        
        # Metrics tracking
        self.metrics = ProcessingMetrics(
            total_time=0,
            agent_times={},
            retry_counts={},
            quality_scores={},
            cache_hits=0,
            cache_misses=0
        )
        
        logger.info("✅ All systems operational")
    
    async def process_chunk_with_review(
        self, 
        chunk_id: int, 
        chunk: str, 
        url: str
    ) -> Dict:
        """
        Process a single chunk with critic review and self-correction loop
        """
        logger.info(f"[Chunk {chunk_id}] Starting intelligent analysis...")
        
        chunk_data = {
            'id': chunk_id,
            'original_text': chunk,
            'summary': '',
            'points': [],
            'insights': [],
            'qna': [],
            'chart_data': None,
            'image_prompt': '',
            'quality_report': {}
        }
        
        # === PHASE 1: Initial Generation (Parallel) ===
        tasks = [
            self.summarizer.summarize_async(chunk),
            self.key_point.extract_points_async(chunk),
            self.insight.find_insights_async(chunk),
            self.qna.generate_qna_async(chunk)
        ]
        
        results = await asyncio.gather(*tasks)
        summary_output, points_output, insights_output, qna_output = results
        
        # Store initial results
        chunk_data['summary'] = summary_output.content
        chunk_data['points'] = points_output.content
        chunk_data['insights'] = insights_output.content
        chunk_data['qna'] = qna_output.content
        
        # Track metrics
        for output in results:
            self.metrics.agent_times[output.agent_name] = \
                self.metrics.agent_times.get(output.agent_name, 0) + output.processing_time
            self.metrics.quality_scores[f"{output.agent_name}_{chunk_id}"] = output.quality_score
        
        # === PHASE 2: Critical Review ===
        logger.info(f"[Chunk {chunk_id}] Performing quality review...")
        
        review_tasks = [
            self.critic.review_summary(chunk_data['summary'], chunk),
            self.critic.review_key_points(chunk_data['points'], chunk),
            self.critic.review_insights(chunk_data['insights'], chunk)
        ]
        
        reviews = await asyncio.gather(*review_tasks)
        summary_review, points_review, insights_review = reviews
        
        # === PHASE 3: Self-Correction (if needed) ===
        needs_correction = False
        
        # Check if summary needs revision
        if summary_review.quality_score.value < QualityScore.ACCEPTABLE.value:
            logger.warning(f"[Chunk {chunk_id}] Summary quality insufficient, regenerating...")
            needs_correction = True
            
            # Regenerate with feedback
            feedback_prompt = f"Previous attempt had issues: {', '.join(summary_review.revision_notes)}"
            revised = await self.summarizer.summarize_async(
                f"{feedback_prompt}\n\nOriginal text: {chunk}"
            )
            chunk_data['summary'] = revised.content
            self.metrics.retry_counts['summarizer'] = \
                self.metrics.retry_counts.get('summarizer', 0) + 1
        
        # Check if insights need revision
        if insights_review.quality_score.value < QualityScore.ACCEPTABLE.value:
            logger.warning(f"[Chunk {chunk_id}] Insights too shallow, regenerating...")
            needs_correction = True
            
            revised = await self.insight.find_insights_async(chunk)
            chunk_data['insights'] = revised.content
            self.metrics.retry_counts['insight'] = \
                self.metrics.retry_counts.get('insight', 0) + 1
        
        # === PHASE 4: Chart & Visual (Non-blocking) ===
        chart_data = self.chart.extract_data(chunk)
        chunk_data['chart_data'] = chart_data
        
        # Generate visual keyword
        visual_kw = self.visual_kw.get_search_term(chunk_data['summary'])
        chunk_data['image_prompt'] = visual_kw
        
        # Store quality report
        chunk_data['quality_report'] = {
            'summary_score': summary_review.quality_score.name,
            'points_score': points_review.quality_score.name,
            'insights_score': insights_review.quality_score.name,
            'corrections_applied': needs_correction
        }
        
        logger.info(f"[Chunk {chunk_id}] ✅ Analysis complete (Quality: {summary_review.quality_score.name})")
        return chunk_data
    
    async def step_1_analyze_async(self, url: str) -> Tuple[List[Dict], RAGEngine]:
        """
        Async analysis pipeline with parallel chunk processing
        """
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("STEP 1: DEEP VIDEO ANALYSIS")
        logger.info("=" * 60)
        
        # Download & Transcribe
        logger.info("📥 Downloading audio...")
        audio_path = download_audio(url)
        
        logger.info("🎤 Transcribing with Whisper...")
        model_size = self.config.get('transcription_model_size', 'base')
        transcript = transcribe_audio_whisper(audio_path, model_size)
        
        if not transcript:
            raise ValueError("Transcription failed or returned empty")
        
        logger.info(f"✅ Transcript: {len(transcript)} characters")
        
        # Chunk text
        chunks = chunk_text(transcript, max_len=2500)
        logger.info(f"📚 Split into {len(chunks)} chunks")
        
        # === PARALLEL CHUNK PROCESSING ===
        logger.info("🚀 Starting parallel analysis of all chunks...")
        
        # Create tasks for all chunks
        chunk_tasks = [
            self.process_chunk_with_review(i, chunk, url)
            for i, chunk in enumerate(chunks)
        ]
        
        # Process all chunks concurrently
        outline = await asyncio.gather(*chunk_tasks)
        
        # === RAG INDEXING ===
        logger.info("📇 Building knowledge base for Q&A...")
        rag = RAGEngine()
        rag.ingest_transcript(chunks)
        
        # Calculate metrics
        self.metrics.total_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info(f"✅ ANALYSIS COMPLETE in {self.metrics.total_time:.1f}s")
        logger.info(f"📊 Quality Scores: {self._summarize_quality()}")
        logger.info(f"🔄 Retries: {sum(self.metrics.retry_counts.values())}")
        logger.info("=" * 60)
        
        return outline, rag
    
    def step_1_analyze(self, url: str) -> Tuple[List[Dict], RAGEngine]:
        """Sync wrapper for async analysis"""
        return asyncio.run(self.step_1_analyze_async(url))
    
    def _summarize_quality(self) -> str:
        """Generate quality summary"""
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
        logger.info("=" * 60)
        logger.info("STEP 2: INTELLIGENT PRESENTATION GENERATION")
        logger.info("=" * 60)
        
        ppt = PPTGenerator()
        img_gen = ImageGenerator() if use_ai_images else None
        
        # === TITLE GENERATION ===
        logger.info("📝 Generating title...")
        full_summary = " ".join(s['summary'] for s in outline[:3])
        title = self.title_gen.generate_title(full_summary)
        logger.info(f"✅ Title: {title}")
        
        # === TITLE SLIDE ===
        ppt.add_title_slide(title, f"AI-Generated Analysis | {len(outline)} Sections")
        
        # === PROCESS EACH SECTION ===
        for section in outline:
            logger.info(f"[Section {section['id']+1}] Building slides...")
            
            # Slide 1: Visual Overview
            if use_ai_images and img_gen:
                logger.info("  🎨 Generating AI artwork...")
                img_url = img_gen.generate(section['image_prompt'], f"sec{section['id']}")
            else:
                logger.info("  📷 Fetching stock photo...")
                img_url = self.visual.get_image_for_topic(section['image_prompt'])
            
            ppt.add_visual_slide(
                f"Section {section['id']+1}: Overview",
                section['summary'],
                img_url
            )
            
            # Slide 2: Combined Analysis (if content is moderate)
            if len(section['points']) <= 4 and len(section['insights']) <= 3:
                ppt.add_analysis_slide(
                    f"Analysis",
                    section['points'],
                    section['insights']
                )
            else:
                # Slide 2a: Key Points
                ppt.add_bullet_slide(
                    f"Key Findings",
                    section['points'],
                    icon="▸"
                )
                # Slide 2b: Deep Insights
                if section['insights']:
                    ppt.add_bullet_slide(
                        f"Strategic Insights",
                        section['insights'],
                        icon="💡"
                    )
            
            # Slide 3: Chart (if data available)
            if section['chart_data']:
                logger.info("  📊 Creating data visualization...")
                chart_path = ppt.create_chart_image(
                    section['chart_data'],
                    f"sec{section['id']}"
                )
                if chart_path:
                    ppt.add_chart_slide(
                        section['chart_data']['title'],
                        chart_path
                    )
            
            # Slide 4: Q&A (if available)
            if section['qna']:
                ppt.add_qna_slide(section['qna'])
        
        # === FINAL SWOT SLIDE ===
        logger.info("📈 Performing executive SWOT analysis...")
        full_text = " ".join(s['original_text'] for s in outline)
        swot_result = await self.swot.analyze_async(full_text[:5000])
        swot_data = swot_result.content
        
        ppt.add_swot_slide(swot_data)
        
        # === SAVE ===
        logger.info("💾 Saving presentation...")
        path = ppt.save(title)
        
        # Cleanup
        if img_gen:
            img_gen.unload_model()
        
        logger.info("=" * 60)
        logger.info(f"✅ PRESENTATION GENERATED: {path}")
        logger.info("=" * 60)
        
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
            'cache_stats': {
                'hits': self.metrics.cache_hits,
                'misses': self.metrics.cache_misses,
                'hit_rate': self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses) 
                    if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else 0
            }
        }