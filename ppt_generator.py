# ppt_generator.py
from pptx import Presentation
from pptx.util import Pt, Inches
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import requests
from io import BytesIO
import logging
from pathlib import Path
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
SLIDE_DIR = Path("outputs/slides")
SLIDE_DIR.mkdir(parents=True, exist_ok=True)

class PPTGenerator:
    def __init__(self):
        self.prs = Presentation()

    def _smart_text_box(self, slide, text, left, top, width, height, max_font=20, is_bold=False):
        """Automatically fits text to box."""
        txBox = slide.shapes.add_textbox(left, top, width, height)
        tf = txBox.text_frame
        tf.word_wrap = True
        
        if len(text) > 900: text = text[:900] + "..."
        
        p = tf.add_paragraph()
        p.text = text
        if is_bold: p.font.bold = True
        
        length = len(text)
        if length < 100: fs = max_font
        elif length < 300: fs = max_font - 2
        elif length < 500: fs = 14
        elif length < 700: fs = 12
        else: fs = 10
        p.font.size = Pt(fs)
        return txBox

    def create_chart_image(self, data: dict, suffix: str):
        try:
            plt.clf()
            plt.figure(figsize=(8, 5)) # Larger chart
            bars = plt.bar(data['labels'], data['values'], color='#2980B9')
            plt.title(data.get('title', 'Data Analysis'), fontsize=14, pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            path = SLIDE_DIR / f"chart_{suffix}.png"
            plt.savefig(path, dpi=150)
            plt.close()
            return str(path)
        except: return None

    def add_visual_slide(self, header, summary, image_url=None):
        """Slide 1: Visual Context"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6]) 
        
        # Header
        shape = slide.shapes.add_shape(1, Inches(0), Inches(0.5), Inches(10), Inches(1))
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor(44, 62, 80)
        tf = shape.text_frame
        tf.text = header
        tf.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
        tf.paragraphs[0].font.size = Pt(28)
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        
        # Summary
        self._smart_text_box(slide, summary, Inches(0.5), Inches(1.8), Inches(4.5), Inches(5), max_font=24)
        
        # Image
        if image_url:
            try:
                resp = requests.get(image_url)
                slide.shapes.add_picture(BytesIO(resp.content), Inches(5.2), Inches(1.8), width=Inches(4.5))
            except: pass

    def add_analysis_slide(self, header, key_points, insights):
        """Combined Layout (for lighter content)"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        
        t = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(1))
        t.text_frame.text = f"Analysis: {header}"
        t.text_frame.paragraphs[0].font.bold = True
        t.text_frame.paragraphs[0].font.size = Pt(24)

        # Left: Facts
        bg_l = slide.shapes.add_shape(1, Inches(0.5), Inches(1.5), Inches(4.4), Inches(5))
        bg_l.fill.solid()
        bg_l.fill.fore_color.rgb = RGBColor(245, 245, 245)
        tf_l = bg_l.text_frame
        p = tf_l.add_paragraph()
        p.text = "📌 Key Facts"
        p.font.bold = True
        p.font.color.rgb = RGBColor(0, 50, 100)
        
        for pt in key_points:
            p = tf_l.add_paragraph()
            p.text = f"• {pt}"
            p.font.size = Pt(14)

        # Right: Insights
        bg_r = slide.shapes.add_shape(1, Inches(5.1), Inches(1.5), Inches(4.4), Inches(5))
        bg_r.fill.solid()
        bg_r.fill.fore_color.rgb = RGBColor(230, 240, 255)
        tf_r = bg_r.text_frame
        p = tf_r.add_paragraph()
        p.text = "💡 Implications"
        p.font.bold = True
        p.font.color.rgb = RGBColor(0, 100, 50)
        
        for ins in insights:
            p = tf_r.add_paragraph()
            p.text = f"► {ins}"
            p.font.size = Pt(14)

    def add_bullet_slide(self, title, items, icon="•"):
        """Generic Bullet Slide (for overflow content)"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[1])
        slide.shapes.title.text = title
        
        tf = slide.placeholders[1].text_frame
        tf.clear()
        
        for item in items:
            p = tf.add_paragraph()
            p.text = f"{icon} {item}"
            p.font.size = Pt(20)
            p.space_after = Pt(12)

    def add_chart_slide(self, header, chart_path, analysis_text=None):
        """Dedicated Chart Slide"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        
        t = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(1))
        t.text_frame.text = header
        t.text_frame.paragraphs[0].font.bold = True
        t.text_frame.paragraphs[0].font.size = Pt(24)
        
        try:
            slide.shapes.add_picture(chart_path, Inches(1), Inches(1.5), width=Inches(8))
        except: pass

    def add_qna_slide(self, qna_list):
        """Dedicated Q&A Slide"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[1])
        slide.shapes.title.text = "Q&A Discussion"
        
        tf = slide.placeholders[1].text_frame
        tf.clear()
        
        for q, a in qna_list:
            p_q = tf.add_paragraph()
            p_q.text = f"Q: {q}"
            p_q.font.bold = True
            p_q.font.color.rgb = RGBColor(0, 0, 150)
            p_q.font.size = Pt(18)
            
            p_a = tf.add_paragraph()
            p_a.text = f"A: {a}"
            p_a.font.size = Pt(16)
            p_a.space_after = Pt(20)

    def add_swot_slide(self, swot_data):
        # (Same SWOT code as before)
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        t = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(1))
        t.text_frame.text = "Executive SWOT Analysis"
        t.text_frame.paragraphs[0].font.size = Pt(32)
        t.text_frame.paragraphs[0].font.bold = True
        
        if not isinstance(swot_data, dict): swot_data = {"S":"", "W":"", "O":"", "T":""}
        def draw_card(label, text, x, y, color):
            shape = slide.shapes.add_shape(1, x, y, Inches(4.5), Inches(2.2))
            shape.fill.solid()
            shape.fill.fore_color.rgb = color
            tf = shape.text_frame
            tf.text = f"{label}\n{text}"
            tf.paragraphs[0].font.bold = True
            if len(text) > 300: tf.paragraphs[1].font.size = Pt(10)
            elif len(text) > 200: tf.paragraphs[1].font.size = Pt(11)
            else: tf.paragraphs[1].font.size = Pt(12)

        draw_card("STRENGTHS", swot_data.get('S',''), Inches(0.5), Inches(1.5), RGBColor(200, 240, 200))
        draw_card("WEAKNESSES", swot_data.get('W',''), Inches(5.1), Inches(1.5), RGBColor(250, 210, 210))
        draw_card("OPPORTUNITIES", swot_data.get('O',''), Inches(0.5), Inches(4.0), RGBColor(210, 230, 255))
        draw_card("THREATS", swot_data.get('T',''), Inches(5.1), Inches(4.0), RGBColor(255, 240, 200))

    def save(self, filename):
        safe = "".join([c for c in filename if c.isalnum() or c==' ']).strip()
        path = SLIDE_DIR / f"{safe}.pptx"
        self.prs.save(path)
        return path