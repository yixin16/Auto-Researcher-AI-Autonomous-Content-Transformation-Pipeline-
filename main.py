# main.py
import yaml
import logging
import os
import re

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["NUMEXPR_MAX_THREADS"] = "16"

from utils import download_audio, transcribe_audio_whisper, chunk_text
from cache_manager import get_or_create
from ppt_generator import PPTGenerator

# Import Agents
from agents import (
    BaseAgent, SummarizerAgent, KeyPointAgent, QnAAgent, InsightAgent,
    ChartAgent, SWOTAgent, VisualKeywordAgent, VisualAgent, TitleAgent
)
from rag_engine import RAGEngine
from image_generator import ImageGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    with open("config.yaml") as f: config = yaml.safe_load(f) or {}
except: config = {}

class ContentOrchestrator:
    def __init__(self, config):
        model_name = config.get('llm_model', 'microsoft/phi-2')
        self.base = BaseAgent(model_name=model_name)
        
        # Initialize Agents
        self.summ = SummarizerAgent(model=self.base.model, tokenizer=self.base.tokenizer)
        self.kp = KeyPointAgent(model=self.base.model, tokenizer=self.base.tokenizer)
        self.insight = InsightAgent(model=self.base.model, tokenizer=self.base.tokenizer)
        self.qna = QnAAgent(model=self.base.model, tokenizer=self.base.tokenizer)
        self.chart = ChartAgent(model=self.base.model, tokenizer=self.base.tokenizer)
        self.visual_kw = VisualKeywordAgent(model=self.base.model, tokenizer=self.base.tokenizer)
        self.swot = SWOTAgent(model=self.base.model, tokenizer=self.base.tokenizer)
        self.title = TitleAgent(model=self.base.model, tokenizer=self.base.tokenizer)
        self.visual_api = VisualAgent()
        
        self.rag = RAGEngine()
        self.img_gen = ImageGenerator()

    def _sanitize_title(self, raw_title):
        if "Final Answer:" in raw_title: raw_title = raw_title.split("Final Answer:")[-1]
        clean = raw_title.split('\n')[0].strip()
        clean = re.sub(r'[^\w\s-]', '', clean)
        return clean[:50].strip() or "Presentation_Deck"

    def step_1_analyze(self, url):
        logger.info("🚀 PHASE 1: ANALYSIS")
        audio_path = get_or_create(url, "audio", download_audio, url)
        transcript = get_or_create(url, "transcribe", transcribe_audio_whisper, audio_path, config.get('transcription_model_size', 'base'))
        chunks = chunk_text(transcript)
        
        self.rag.ingest_transcript(chunks)
        
        outline = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Analyzing Section {i+1}/{len(chunks)}...")
            summary = get_or_create(url, f"sum_{i}", self.summ.summarize, chunk)
            points = get_or_create(url, f"pts_{i}", self.kp.extract_points, chunk)
            insights = get_or_create(url, f"ins_{i}", self.insight.find_insights, chunk)
            img_prompt = get_or_create(url, f"img_term_{i}", self.visual_kw.get_search_term, summary)
            
            outline.append({
                "id": i,
                "original_text": chunk,
                "summary": summary,
                "points": points,
                "insights": insights,
                "image_prompt": img_prompt
            })
        return outline, self.rag

    def step_2_generate(self, outline_data, url, use_ai_images=False):
        logger.info("🚀 PHASE 2: GENERATION (Dynamic Mode)")
        ppt = PPTGenerator()
        combined_summary = ""

        for slide in outline_data:
            i = slide['id']
            logger.info(f"Building Slide Set {i+1}...")
            combined_summary += slide['summary'] + " "

            # 1. Get Image
            img_path = None
            if use_ai_images:
                img_path = self.img_gen.generate(slide['image_prompt'], f"slide_{i}")
                self.img_gen.unload_model()
            else:
                img_path = self.visual_api.get_image_for_topic(slide['image_prompt'])

            # 2. Get Chart Data
            chart_data = self.chart.extract_data(slide['original_text'])
            chart_img = None
            if chart_data and isinstance(chart_data, dict):
                chart_img = ppt.create_chart_image(chart_data, f"chart_{i}")
            
            # 3. Get QnA
            qna = self.qna.generate_qna(slide['original_text'])
            
            # --- DYNAMIC SLIDE GENERATION LOGIC ---
            
            # Slide 1: Always Overview
            ppt.add_visual_slide(f"Section {i+1}: Overview", slide['summary'], img_path)
            
            # Logic: Check Content Density
            points = slide['points']
            insights = slide.get('insights', [])
            total_items = len(points) + len(insights)
            
            # Slide 2: Analysis
            if total_items > 6:
                # Content is HEAVY -> Split into two slides
                if points:
                    ppt.add_bullet_slide("Key Takeaways", points, icon="•")
                if insights:
                    ppt.add_bullet_slide("Strategic Insights", insights, icon="►")
            else:
                # Content is LIGHT -> Combine into one split slide
                ppt.add_analysis_slide("Deep Dive", points, insights)

            # Slide 3: Data (Only if chart exists)
            if chart_img:
                ppt.add_chart_slide("Data Analysis", chart_img)

            # Slide 4: Q&A (Always good for engagement)
            if qna:
                ppt.add_qna_slide(qna)

        # Global Analysis
        swot = self.swot.analyze(combined_summary[:2000])
        if isinstance(swot, str): swot = {"S": swot, "W": "", "O": "", "T": ""}
        ppt.add_swot_slide(swot)
        
        raw_title = self.title.generate_title(combined_summary[:1000])
        final_title = self._sanitize_title(raw_title)
        output_path = ppt.save(final_title)
        
        return output_path, final_title, swot

if __name__ == "__main__":
    pass