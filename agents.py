# agents.py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import logging
import os
import re
import json
from pexels_api import API
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

class BaseAgent:
    def __init__(self, model=None, tokenizer=None, model_name="microsoft/phi-2"):
        if model and tokenizer:
            self.model = model
            self.tokenizer = tokenizer
        else:
            logger.info(f"Loading Intelligence Engine ({model_name})...")
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

    def generate(self, prompt, max_new_tokens=512, temperature=0.9):
        formatted_prompt = f"Instruct: {prompt}\nOutput:"
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        try:
            out = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=True, 
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
            output_text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            if "Output:" in output_text:
                return output_text.split("Output:")[1].strip()
            return output_text.replace(formatted_prompt, "").strip()
        except Exception as e:
            logger.error(f"Generation Error: {e}")
            return ""

class SummarizerAgent(BaseAgent):
    def summarize(self, text_chunk):
        prompt = f"Summarize the following video transcript section into a concise, professional executive summary (max 3 sentences). Focus on the core message:\n\n{text_chunk}"
        return self.generate(prompt, max_new_tokens=150, temperature=0.6)

class KeyPointAgent(BaseAgent):
    def extract_points(self, text):
        prompt = f"Extract exactly 4 factual key takeaways from this text. Keep them short and direct (max 10 words each):\n\n{text}"
        out = self.generate(prompt, max_new_tokens=200)
        # Clean list
        return [line.strip("- ").strip() for line in out.splitlines() if line.strip() and len(line) > 5][:4]

class InsightAgent(BaseAgent):
    """Extracts deep, non-obvious analysis."""
    def find_insights(self, text):
        prompt = f"Analyze this text for 'Hidden Insights' or 'Strategic Implications'. Provide 3 points that explain WHY this matters or what the future impact is:\n\n{text}"
        out = self.generate(prompt, max_new_tokens=250, temperature=1.0)
        return [line.strip("- ").strip() for line in out.splitlines() if line.strip() and len(line) > 10][:3]

class QnAAgent(BaseAgent):
    def generate_qna(self, text):
        prompt = f"Create 2 insightful Q&A pairs based on this text. \nFormat:\nQ: [Question]\nA: [Answer]\n\nText: {text}"
        out = self.generate(prompt, max_new_tokens=250)
        qna = []
        curr_q = ""
        for line in out.splitlines():
            if line.startswith("Q:"): curr_q = line.replace("Q:", "").strip()
            elif line.startswith("A:") and curr_q:
                qna.append((curr_q, line.replace("A:", "").strip()))
                curr_q = ""
        return qna

class ChartAgent(BaseAgent):
    def extract_data(self, text):
        prompt = f"""Does this text contain specific numbers comparing items or trends? 
If YES, output ONLY JSON: {{"title": "Chart Title", "labels": ["A","B"], "values": [10, 20]}}
If NO, output: NO_DATA
Text: {text}"""
        res = self.generate(prompt, max_new_tokens=200, temperature=0.2)
        if "NO_DATA" in res: return None
        try:
            json_str = re.search(r'\{.*\}', res, re.DOTALL).group()
            return json.loads(json_str)
        except: return None

class SWOTAgent(BaseAgent):
    def analyze(self, text):
        prompt = f"Perform a SWOT Analysis (Strengths, Weaknesses, Opportunities, Threats) on this content.\nFormat:\nS: ...\nW: ...\nO: ...\nT: ...\n\nText: {text}"
        out = self.generate(prompt, max_new_tokens=400)
        swot = {"S": "N/A", "W": "N/A", "O": "N/A", "T": "N/A"}
        for line in out.splitlines():
            if line.startswith("S:"): swot["S"] = line.replace("S:", "").strip()
            elif line.startswith("W:"): swot["W"] = line.replace("W:", "").strip()
            elif line.startswith("O:"): swot["O"] = line.replace("O:", "").strip()
            elif line.startswith("T:"): swot["T"] = line.replace("T:", "").strip()
        return swot

class VisualKeywordAgent(BaseAgent):
    def get_search_term(self, text):
        prompt = f"Give ONE single noun (visual object) to represent this text. Example: 'Technology', 'Meeting', 'Graph'.\nText: {text[:150]}"
        return self.generate(prompt, max_new_tokens=10).strip()

class TitleAgent(BaseAgent):
    def generate_title(self, summary_text):
        prompt = f"Create a short presentation title (max 5 words). Do not output thoughts:\n\n{summary_text}"
        return self.generate(prompt, max_new_tokens=20).strip().replace('"','')

class VisualAgent:
    def __init__(self):
        self.api_key = PEXELS_API_KEY
        if not self.api_key: self.api = None
        else: self.api = API(self.api_key)
    def get_image_for_topic(self, topic):
        if not self.api: return None
        try:
            self.api.search(topic.split()[0], page=1, results_per_page=1)
            if self.api.get_entries(): return self.api.get_entries()[0].original
        except: return None