from pptx import Presentation
from pptx.util import Pt, Inches, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import requests
from io import BytesIO
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)
SLIDE_DIR = Path("outputs/slides")
SLIDE_DIR.mkdir(parents=True, exist_ok=True)


class DesignTheme:
    """Professional color schemes and design constants"""
    
    # Modern Corporate Theme
    PRIMARY = RGBColor(25, 52, 65)        # Deep Blue
    SECONDARY = RGBColor(235, 94, 40)     # Vibrant Orange
    ACCENT = RGBColor(0, 156, 166)        # Teal
    LIGHT_BG = RGBColor(245, 247, 250)    # Light Gray
    TEXT_DARK = RGBColor(33, 37, 41)      # Almost Black
    TEXT_LIGHT = RGBColor(108, 117, 125)  # Medium Gray
    
    # Typography
    TITLE_FONT = "Calibri Light"
    BODY_FONT = "Calibri"
    ACCENT_FONT = "Calibri Light"
    
    # Spacing
    MARGIN = Inches(0.5)
    PADDING = Inches(0.2)


class PPTGenerator:
    """Enhanced PowerPoint generator with professional design"""
    
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = Inches(10)
        self.prs.slide_height = Inches(7.5)
        self.theme = DesignTheme()
        self.slide_counter = 0
    
    def _add_footer(self, slide, text: str = None):
        """Add consistent footer to slides"""
        footer_height = Inches(0.4)
        footer = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            0, self.prs.slide_height - footer_height,
            self.prs.slide_width, footer_height
        )
        footer.fill.solid()
        footer.fill.fore_color.rgb = self.theme.PRIMARY
        footer.line.color.rgb = self.theme.PRIMARY
        
        if text:
            tf = footer.text_frame
            tf.text = text
            tf.paragraphs[0].font.size = Pt(10)
            tf.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
            tf.paragraphs[0].alignment = PP_ALIGN.RIGHT
    
    def _add_header_bar(self, slide, title: str, color: RGBColor = None):
        """Add modern header bar"""
        color = color or self.theme.PRIMARY
        
        header = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            0, 0,
            self.prs.slide_width, Inches(1.2)
        )
        header.fill.solid()
        header.fill.fore_color.rgb = color
        header.line.color.rgb = color
        
        # Add accent line
        accent_line = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            0, Inches(1.15),
            self.prs.slide_width, Inches(0.05)
        )
        accent_line.fill.solid()
        accent_line.fill.fore_color.rgb = self.theme.SECONDARY
        accent_line.line.width = 0
        
        # Title text
        title_box = slide.shapes.add_textbox(
            self.theme.MARGIN, Inches(0.3),
            self.prs.slide_width - self.theme.MARGIN * 2, Inches(0.8)
        )
        tf = title_box.text_frame
        tf.text = title
        tf.paragraphs[0].font.name = self.theme.TITLE_FONT
        tf.paragraphs[0].font.size = Pt(32)
        tf.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
        tf.paragraphs[0].font.bold = True
    
    def _smart_text_box(
        self, 
        slide, 
        text: str, 
        left: Emu, 
        top: Emu, 
        width: Emu, 
        height: Emu,
        font_size: int = None,
        is_bold: bool = False,
        color: RGBColor = None,
        alignment: PP_ALIGN = PP_ALIGN.LEFT
    ):
        """Enhanced text box with automatic sizing"""
        txBox = slide.shapes.add_textbox(left, top, width, height)
        tf = txBox.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.TOP
        
        # Truncate if too long
        if len(text) > 1200:
            text = text[:1200] + "..."
        
        p = tf.add_paragraph()
        p.text = text
        p.font.name = self.theme.BODY_FONT
        p.alignment = alignment
        
        if is_bold:
            p.font.bold = True
        
        if color:
            p.font.color.rgb = color
        else:
            p.font.color.rgb = self.theme.TEXT_DARK
        
        # Automatic font sizing
        if font_size:
            p.font.size = Pt(font_size)
        else:
            length = len(text)
            if length < 100:
                p.font.size = Pt(20)
            elif length < 300:
                p.font.size = Pt(16)
            elif length < 500:
                p.font.size = Pt(14)
            elif length < 800:
                p.font.size = Pt(12)
            else:
                p.font.size = Pt(11)
        
        return txBox
    
    def add_title_slide(self, title: str, subtitle: str = ""):
        """Professional title slide with modern design"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        
        # Background gradient effect (simulated with shapes)
        bg1 = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            0, 0,
            self.prs.slide_width, self.prs.slide_height
        )
        bg1.fill.solid()
        bg1.fill.fore_color.rgb = self.theme.PRIMARY
        bg1.line.width = 0
        
        # Accent shape
        accent = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(0.5), Inches(2),
            Inches(9), Inches(3)
        )
        accent.fill.solid()
        accent.fill.fore_color.rgb = RGBColor(255, 255, 255)
        accent.line.width = 0
        accent.shadow.inherit = False
        
        # Title
        title_box = slide.shapes.add_textbox(
            Inches(1), Inches(2.3),
            Inches(8), Inches(1.5)
        )
        tf = title_box.text_frame
        tf.text = title
        tf.paragraphs[0].font.name = self.theme.TITLE_FONT
        tf.paragraphs[0].font.size = Pt(44)
        tf.paragraphs[0].font.bold = True
        tf.paragraphs[0].font.color.rgb = self.theme.PRIMARY
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        
        # Subtitle
        if subtitle:
            subtitle_box = slide.shapes.add_textbox(
                Inches(1), Inches(4),
                Inches(8), Inches(0.8)
            )
            tf = subtitle_box.text_frame
            tf.text = subtitle
            tf.paragraphs[0].font.name = self.theme.BODY_FONT
            tf.paragraphs[0].font.size = Pt(18)
            tf.paragraphs[0].font.color.rgb = self.theme.TEXT_LIGHT
            tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        
        # Date/Generated info
        import datetime
        date_text = f"Generated: {datetime.datetime.now().strftime('%B %d, %Y')}"
        date_box = slide.shapes.add_textbox(
            Inches(3), Inches(6.5),
            Inches(4), Inches(0.5)
        )
        tf = date_box.text_frame
        tf.text = date_text
        tf.paragraphs[0].font.size = Pt(12)
        tf.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        
        self.slide_counter += 1
    
    def add_visual_slide(
        self, 
        header: str, 
        summary: str, 
        image_url: Optional[str] = None
    ):
        """Enhanced visual context slide with image"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        
        # Header
        self._add_header_bar(slide, header)
        
        # Content area with background
        content_bg = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            self.theme.MARGIN, Inches(1.5),
            Inches(4.5), Inches(5)
        )
        content_bg.fill.solid()
        content_bg.fill.fore_color.rgb = self.theme.LIGHT_BG
        content_bg.line.width = Pt(1)
        content_bg.line.color.rgb = RGBColor(200, 200, 200)
        
        # Summary text
        self._smart_text_box(
            slide, summary,
            Inches(0.7), Inches(1.8),
            Inches(4), Inches(4.5),
            font_size=18
        )
        
        # Image
        if image_url:
            try:
                resp = requests.get(image_url, timeout=10)
                img_stream = BytesIO(resp.content)
                
                # Add image with border
                pic = slide.shapes.add_picture(
                    img_stream,
                    Inches(5.2), Inches(1.5),
                    width=Inches(4.3)
                )
                
                # Add subtle border
                pic.line.color.rgb = RGBColor(200, 200, 200)
                pic.line.width = Pt(1)
            except Exception as e:
                logger.error(f"Failed to add image: {e}")
        
        self._add_footer(slide, f"Slide {self.slide_counter}")
        self.slide_counter += 1
    
    def add_analysis_slide(
        self, 
        header: str, 
        key_points: List[str], 
        insights: List[str]
    ):
        """Two-column analysis layout"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        
        self._add_header_bar(slide, f"Analysis: {header}", self.theme.ACCENT)
        
        # Left column: Key Facts
        left_bg = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            self.theme.MARGIN, Inches(1.5),
            Inches(4.5), Inches(5.2)
        )
        left_bg.fill.solid()
        left_bg.fill.fore_color.rgb = RGBColor(240, 248, 255)  # Light blue
        left_bg.line.width = Pt(2)
        left_bg.line.color.rgb = self.theme.ACCENT
        
        # Header for facts
        fact_header = slide.shapes.add_textbox(
            Inches(0.7), Inches(1.7),
            Inches(4), Inches(0.4)
        )
        tf = fact_header.text_frame
        tf.text = "📌 KEY FACTS"
        tf.paragraphs[0].font.name = self.theme.ACCENT_FONT
        tf.paragraphs[0].font.size = Pt(20)
        tf.paragraphs[0].font.bold = True
        tf.paragraphs[0].font.color.rgb = self.theme.ACCENT
        
        # Facts content
        facts_box = slide.shapes.add_textbox(
            Inches(0.7), Inches(2.2),
            Inches(4), Inches(4.2)
        )
        tf = facts_box.text_frame
        tf.word_wrap = True
        
        for i, point in enumerate(key_points[:5]):
            p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
            p.text = f"▸ {point}"
            p.font.name = self.theme.BODY_FONT
            p.font.size = Pt(14)
            p.font.color.rgb = self.theme.TEXT_DARK
            p.space_after = Pt(10)
            p.level = 0
        
        # Right column: Strategic Insights
        right_bg = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(5.1), Inches(1.5),
            Inches(4.5), Inches(5.2)
        )
        right_bg.fill.solid()
        right_bg.fill.fore_color.rgb = RGBColor(255, 248, 240)  # Light orange
        right_bg.line.width = Pt(2)
        right_bg.line.color.rgb = self.theme.SECONDARY
        
        # Header for insights
        insight_header = slide.shapes.add_textbox(
            Inches(5.3), Inches(1.7),
            Inches(4), Inches(0.4)
        )
        tf = insight_header.text_frame
        tf.text = "💡 STRATEGIC IMPLICATIONS"
        tf.paragraphs[0].font.name = self.theme.ACCENT_FONT
        tf.paragraphs[0].font.size = Pt(18)
        tf.paragraphs[0].font.bold = True
        tf.paragraphs[0].font.color.rgb = self.theme.SECONDARY
        
        # Insights content
        insights_box = slide.shapes.add_textbox(
            Inches(5.3), Inches(2.2),
            Inches(4), Inches(4.2)
        )
        tf = insights_box.text_frame
        tf.word_wrap = True
        
        for i, insight in enumerate(insights[:4]):
            p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
            p.text = f"→ {insight}"
            p.font.name = self.theme.BODY_FONT
            p.font.size = Pt(13)
            p.font.color.rgb = self.theme.TEXT_DARK
            p.space_after = Pt(12)
            p.level = 0
        
        self._add_footer(slide, f"Slide {self.slide_counter}")
        self.slide_counter += 1
    
    def add_bullet_slide(
        self, 
        title: str, 
        items: List[str], 
        icon: str = "▸"
    ):
        """Clean bullet point slide"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        
        self._add_header_bar(slide, title)
        
        # Content box
        content_box = slide.shapes.add_textbox(
            Inches(1), Inches(1.8),
            Inches(8), Inches(5)
        )
        tf = content_box.text_frame
        tf.word_wrap = True
        
        for i, item in enumerate(items):
            p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
            p.text = f"{icon} {item}"
            p.font.name = self.theme.BODY_FONT
            p.font.size = Pt(16)
            p.font.color.rgb = self.theme.TEXT_DARK
            p.space_after = Pt(18)
            p.level = 0
        
        self._add_footer(slide, f"Slide {self.slide_counter}")
        self.slide_counter += 1
    
    def create_chart_image(self, data: Dict, suffix: str) -> Optional[str]:
        """Create professional data visualization"""
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = ['#00ADD8', '#FF5722', '#4CAF50', '#FFC107', '#9C27B0']
            
            bars = ax.bar(
                data['labels'], 
                data['values'],
                color=colors[:len(data['labels'])],
                edgecolor='white',
                linewidth=2
            )
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold'
                )
            
            ax.set_title(
                data.get('title', 'Data Analysis'),
                fontsize=16, fontweight='bold', pad=20
            )
            ax.set_xlabel('Categories', fontsize=12, fontweight='bold')
            ax.set_ylabel('Values', fontsize=12, fontweight='bold')
            
            plt.xticks(rotation=30, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            
            path = SLIDE_DIR / f"chart_{suffix}.png"
            plt.savefig(path, dpi=200, bbox_inches='tight')
            plt.close()
            
            return str(path)
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return None
    
    def add_chart_slide(self, header: str, chart_path: str):
        """Dedicated chart visualization slide"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        
        self._add_header_bar(slide, f"📊 {header}", self.theme.ACCENT)
        
        try:
            slide.shapes.add_picture(
                chart_path,
                Inches(0.5), Inches(1.7),
                width=Inches(9)
            )
        except Exception as e:
            logger.error(f"Failed to add chart: {e}")
        
        self._add_footer(slide, f"Slide {self.slide_counter}")
        self.slide_counter += 1
    
    def add_qna_slide(self, qna_list: List[Tuple[str, str]]):
        """Interactive Q&A slide"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        
        self._add_header_bar(slide, "💬 Discussion & Q&A")
        
        y_offset = Inches(1.8)
        
        for q, a in qna_list[:3]:
            # Question box
            q_bg = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                Inches(0.7), y_offset,
                Inches(8.6), Inches(0.6)
            )
            q_bg.fill.solid()
            q_bg.fill.fore_color.rgb = self.theme.ACCENT
            q_bg.line.width = 0
            
            q_text = slide.shapes.add_textbox(
                Inches(1), y_offset + Inches(0.1),
                Inches(8), Inches(0.4)
            )
            tf = q_text.text_frame
            tf.text = f"Q: {q}"
            tf.paragraphs[0].font.name = self.theme.BODY_FONT
            tf.paragraphs[0].font.size = Pt(14)
            tf.paragraphs[0].font.bold = True
            tf.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
            
            y_offset += Inches(0.7)
            
            # Answer box
            a_text = slide.shapes.add_textbox(
                Inches(1.2), y_offset,
                Inches(7.8), Inches(0.8)
            )
            tf = a_text.text_frame
            tf.text = f"A: {a}"
            tf.word_wrap = True
            tf.paragraphs[0].font.name = self.theme.BODY_FONT
            tf.paragraphs[0].font.size = Pt(13)
            tf.paragraphs[0].font.color.rgb = self.theme.TEXT_DARK
            
            y_offset += Inches(1)
        
        self._add_footer(slide, f"Slide {self.slide_counter}")
        self.slide_counter += 1
    
    def add_swot_slide(self, swot_data: Dict):
        """Professional SWOT analysis slide"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        
        self._add_header_bar(slide, "Executive SWOT Analysis", self.theme.PRIMARY)
        
        # Define quadrant positions
        box_width = Inches(4.5)
        box_height = Inches(2.5)
        margin = Inches(0.5)
        gap = Inches(0.2)
        
        # Helper to create SWOT box
        def create_swot_box(label, text, left, top, color, icon):
            # Background
            bg = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                left, top, box_width, box_height
            )
            bg.fill.solid()
            bg.fill.fore_color.rgb = color
            bg.line.width = Pt(2)
            bg.line.color.rgb = RGBColor(255, 255, 255)
            
            # Label
            label_box = slide.shapes.add_textbox(
                left + Inches(0.2), top + Inches(0.15),
                box_width - Inches(0.4), Inches(0.4)
            )
            tf = label_box.text_frame
            tf.text = f"{icon} {label}"
            tf.paragraphs[0].font.name = self.theme.ACCENT_FONT
            tf.paragraphs[0].font.size = Pt(18)
            tf.paragraphs[0].font.bold = True
            tf.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
            
            # Content
            content_box = slide.shapes.add_textbox(
                left + Inches(0.2), top + Inches(0.65),
                box_width - Inches(0.4), box_height - Inches(0.8)
            )
            tf = content_box.text_frame
            tf.text = text
            tf.word_wrap = True
            tf.paragraphs[0].font.name = self.theme.BODY_FONT
            tf.paragraphs[0].font.size = Pt(11)
            tf.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
        
        # Create all 4 quadrants
        create_swot_box(
            "STRENGTHS", swot_data.get('S', 'N/A'),
            margin, Inches(1.7),
            RGBColor(76, 175, 80), "✓"  # Green
        )
        
        create_swot_box(
            "WEAKNESSES", swot_data.get('W', 'N/A'),
            margin + box_width + gap, Inches(1.7),
            RGBColor(244, 67, 54), "✗"  # Red
        )
        
        create_swot_box(
            "OPPORTUNITIES", swot_data.get('O', 'N/A'),
            margin, Inches(1.7) + box_height + gap,
            RGBColor(33, 150, 243), "⬆"  # Blue
        )
        
        create_swot_box(
            "THREATS", swot_data.get('T', 'N/A'),
            margin + box_width + gap, Inches(1.7) + box_height + gap,
            RGBColor(255, 152, 0), "⚠"  # Orange
        )
        
        self._add_footer(slide, f"Slide {self.slide_counter}")
        self.slide_counter += 1
    
    def save(self, filename: str) -> Path:
        """Save presentation with safe filename"""
        import re
        safe = re.sub(r'[<>:"/\\|?*]', '', filename)
        safe = safe[:100]  # Limit length
        path = SLIDE_DIR / f"{safe}.pptx"
        self.prs.save(path)
        logger.info(f"💾 Saved presentation: {path}")
        return path