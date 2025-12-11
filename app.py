import streamlit as st
import os
import time
import torch
import shutil
import asyncio
from dotenv import load_dotenv
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path

# Import Enhanced Orchestrator
try:
    from main import ContentOrchestrator
except ImportError:
    st.error("⚠️ Could not import 'ContentOrchestrator' from 'main.py'. Please ensure the file exists.")
    st.stop()

load_dotenv()

# === PAGE CONFIG ===
st.set_page_config(
    page_title="AutoResearcher AI Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === ENHANCED CSS ===
st.markdown("""
<style>
    /* Modern Dark Theme Adjustment */
    .stApp { background-color: #0E1117; }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        margin: 5px 0;
    }
    
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Progress Box */
    .progress-box {
        background-color: #262730;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    
    /* Chat Bubbles */
    .stChatMessage {
        background-color: #262730;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# === SESSION STATE ===
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "outline" not in st.session_state:
    st.session_state.outline = None
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_url" not in st.session_state:
    st.session_state.current_url = ""
if "performance_metrics" not in st.session_state:
    st.session_state.performance_metrics = None
if "processing_logs" not in st.session_state:
    st.session_state.processing_logs = []

# New State variables for file persistence
if "generated_ppt_path" not in st.session_state:
    st.session_state.generated_ppt_path = None
if "generated_ppt_title" not in st.session_state:
    st.session_state.generated_ppt_title = None
if "generated_swot" not in st.session_state:
    st.session_state.generated_swot = None

# === SIDEBAR ===
with st.sidebar:
    st.title("🧠 AutoResearcher")
    st.caption("v2.1 | Enhanced Multi-Agent System")
    
    # Hardware Status
    st.subheader("🖥️ Hardware Status")
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            vram_used = torch.cuda.memory_allocated(0) / 1024**3
            
            st.success(f"**GPU Active:** {gpu_name}")
            st.progress(min(vram_used / vram_total, 1.0))
            st.caption(f"VRAM: {vram_used:.1f}GB / {vram_total:.1f}GB used")
        elif torch.backends.mps.is_available():
            st.success("**MPS (Apple Silicon) Active**")
        else:
            st.warning("**Running on CPU**")
            st.caption("Performance will be slower")
    except Exception:
        st.error("Hardware check failed")
    
    st.divider()
    
    # Model Configuration
    st.subheader("🤖 Configuration")
    
    model_choice = st.selectbox(
        "Reasoning Model",
        ["microsoft/phi-2", "unsloth/llama-3-8b-Instruct-bnb-4bit"],
        help="Phi-2: Fast, lower memory.\nLlama-3: Higher intelligence, more memory."
    )
    
    whisper_size = st.selectbox(
        "Transcription Model",
        ["base", "small", "medium", "large"],
        index=1,
        help="Balance speed vs accuracy."
    )
    
    st.divider()
    
    # Advanced Features
    with st.expander("🛠️ Advanced Settings"):
        enable_critic = st.toggle(
            "Quality Control Agent",
            value=True,
            help="Enable self-correction loops"
        )
        
        enable_ai_art = st.toggle(
            "AI Image Generation",
            value=False,
            help="Generate custom images (requires GPU)"
        )
        
        parallel_processing = st.toggle(
            "Async Processing",
            value=True,
            help="Process chunks in parallel"
        )
    
    st.divider()
    
    # System Actions
    if st.session_state.orchestrator is None:
        if st.button("🚀 Initialize System", type="primary", use_container_width=True):
            with st.spinner("Booting up AI agents..."):
                try:
                    config = {
                        'llm_model': model_choice,
                        'transcription_model_size': whisper_size,
                        'enable_critic': enable_critic,
                        'parallel_processing': parallel_processing
                    }
                    st.session_state.orchestrator = ContentOrchestrator(config)
                    st.success("✅ System Online")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Initialization failed: {e}")
                    st.exception(e)
    else:
        st.success("🟢 **System Active**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("♻️ Restart", use_container_width=True):
                # Reset all state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                cache_dir = Path("outputs/cache")
                if cache_dir.exists():
                    try:
                        shutil.rmtree(cache_dir)
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        st.success("Cache cleared")
                    except Exception as e:
                        st.error(f"Error: {e}")

# === MAIN CONTENT ===
st.markdown("## 🧠 AutoResearcher AI Pro")
st.markdown("##### Deep Video Analysis with Self-Correcting Intelligence")

# === TAB LAYOUT ===
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Analyze", 
    "📝 Review & Edit", 
    "💬 Q&A Chat",
    "📊 Performance"
])

# === TAB 1: ANALYZE ===
with tab1:
    st.markdown("### 🎯 Video Analysis Pipeline")
    
    col_url, col_btn = st.columns([5, 1])
    with col_url:
        url_input = st.text_input(
            "YouTube URL",
            placeholder="Paste YouTube URL here...",
            label_visibility="collapsed"
        )
    with col_btn:
        analyze_btn = st.button(
            "▶ Analyze",
            type="primary",
            use_container_width=True
        )
    
    if analyze_btn:
        if not st.session_state.orchestrator:
            st.error("⚠️ Please initialize the system first (see sidebar)")
        elif not url_input:
            st.warning("⚠️ Please provide a YouTube URL")
        else:
            st.session_state.current_url = url_input
            st.session_state.processing_logs = []
            
            # Reset previous results
            st.session_state.generated_ppt_path = None
            st.session_state.generated_swot = None
            
            # Progress UI
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            try:
                with st.spinner("🚀 Launching analysis pipeline..."):
                    start_time = time.time()
                    
                    status_placeholder.markdown('<div class="progress-box">📥 <b>Phase 1: Downloading & Transcribing...</b></div>', unsafe_allow_html=True)
                    progress_placeholder.progress(10)
                    
                    # 1. Orchestrator Analysis
                    status_placeholder.markdown('<div class="progress-box">🧠 <b>Phase 2: Agentic Analysis & Knowledge Graph...</b></div>', unsafe_allow_html=True)
                    progress_placeholder.progress(40)
                    
                    # Call orchestrator
                    outline, rag = st.session_state.orchestrator.step_1_analyze(url_input)
                    
                    progress_placeholder.progress(80)
                    status_placeholder.markdown('<div class="progress-box">📚 <b>Phase 3: Finalizing Knowledge Base...</b></div>', unsafe_allow_html=True)
                    
                    st.session_state.outline = outline
                    st.session_state.rag_engine = rag
                    
                    # Get performance metrics
                    st.session_state.performance_metrics = \
                        st.session_state.orchestrator.get_performance_report()
                    
                    progress_placeholder.progress(100)
                    elapsed = time.time() - start_time
                    
                    # Success Display
                    status_placeholder.success(f"✅ **Analysis Complete** in {elapsed:.1f}s")
                    
                    # Display quick stats
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Sections</div>
                            <div class="metric-value">{len(outline)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        total_points = sum(len(s.get('points', [])) for s in outline)
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Key Points</div>
                            <div class="metric-value">{total_points}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        total_insights = sum(len(s.get('insights', [])) for s in outline)
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Insights</div>
                            <div class="metric-value">{total_insights}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        metrics = st.session_state.performance_metrics
                        retries = sum(metrics.get('retry_counts', {}).values())
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Self-Corrections</div>
                            <div class="metric-value">{retries}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.balloons()
                    st.info("👉 **Next Step:** Go to the 'Review & Edit' tab to refine content.")
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                import traceback
                with st.expander("Debug Info"):
                    st.code(traceback.format_exc())

# === TAB 2: REVIEW & EDIT ===
with tab2:
    if st.session_state.outline is None:
        st.info("👈 Please run an analysis in the 'Analyze' tab first")
    else:
        st.markdown("### ✏️ Human-in-the-Loop Editor")
        
        if st.session_state.performance_metrics:
            metrics = st.session_state.performance_metrics
            quality_summary = metrics.get('quality_summary', 'N/A')
            st.caption(f"🤖 AI Confidence Score: {quality_summary}")
        
        # FORM START
        with st.form("slide_editor_v2"):
            updated_outline = []
            
            for i, slide in enumerate(st.session_state.outline):
                with st.expander(
                    f"📄 Section {slide.get('id', i)+1}: {slide.get('summary', '')[:60]}...",
                    expanded=(i == 0)
                ):
                    # Quality indicator
                    quality_report = slide.get('quality_report', {})
                    if quality_report:
                        cols = st.columns(3)
                        cols[0].caption(f"Summary Quality: {quality_report.get('summary_score', 'N/A')}")
                        cols[1].caption(f"Points Quality: {quality_report.get('points_score', 'N/A')}")
                        cols[2].caption(f"Insights Quality: {quality_report.get('insights_score', 'N/A')}")
                    
                    # Content editing
                    c1, c2 = st.columns([3, 1])
                    
                    with c1:
                        new_sum = st.text_area(
                            "Executive Summary",
                            slide.get('summary', ''),
                            height=80,
                            key=f"sum_v2_{i}",
                            help="This appears on the overview slide"
                        )
                    
                    with c2:
                        new_img = st.text_input(
                            "Visual Keyword",
                            slide.get('image_prompt', ''),
                            key=f"img_v2_{i}",
                            help="Used to find stock photos"
                        )
                    
                    c3, c4 = st.columns(2)
                    
                    with c3:
                        points_text = "\n".join(slide.get('points', []))
                        new_pts = st.text_area(
                            "Key Facts (Bullets)",
                            points_text,
                            height=140,
                            key=f"pts_v2_{i}",
                            help="One point per line"
                        )
                    
                    with c4:
                        insights_text = "\n".join(slide.get('insights', []))
                        new_ins = st.text_area(
                            "Strategic Insights",
                            insights_text,
                            height=140,
                            key=f"ins_v2_{i}",
                            help="Deep analysis & implications"
                        )
                    
                    updated_outline.append({
                        "id": slide.get('id', i),
                        "summary": new_sum,
                        "points": [p.strip() for p in new_pts.split("\n") if p.strip()],
                        "insights": [i.strip() for i in new_ins.split("\n") if i.strip()],
                        "image_prompt": new_img,
                        "original_text": slide.get('original_text', ''),
                        "quality_report": quality_report,
                        "chart_data": slide.get('chart_data'),
                        "qna": slide.get('qna')
                    })
            
            st.divider()
            
            col_gen1, col_gen2, col_gen3 = st.columns([2, 1, 2])
            with col_gen2:
                generate_btn = st.form_submit_button(
                    "🎬 Generate Deck",
                    type="primary",
                    use_container_width=True
                )
        # FORM END
        
        # LOGIC OUTSIDE FORM
        if generate_btn:
            with st.spinner("🏗️ Building your presentation..."):
                try:
                    path, title, swot = st.session_state.orchestrator.step_2_generate(
                        updated_outline,
                        st.session_state.current_url,
                        use_ai_images=enable_ai_art
                    )
                    
                    # Store results in session state to persist after rerun
                    st.session_state.generated_ppt_path = path
                    st.session_state.generated_ppt_title = title
                    st.session_state.generated_swot = swot
                    
                    st.success(f"✅ **Presentation Generated:** {title}")
                    
                except Exception as e:
                    st.error(f"❌ Generation failed: {str(e)}")
                    st.exception(e)

        # DOWNLOAD SECTION (Visible if generation has happened)
        if st.session_state.generated_ppt_path and os.path.exists(st.session_state.generated_ppt_path):
            st.divider()
            st.markdown("### 📥 Download")
            
            with open(st.session_state.generated_ppt_path, "rb") as f:
                st.download_button(
                    label=f"Download {st.session_state.generated_ppt_title}.pptx",
                    data=f,
                    file_name=f"{st.session_state.generated_ppt_title}.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True,
                    key="download_ppt_btn"
                )
            
            # SWOT Display
            if st.session_state.generated_swot:
                st.subheader("📊 Executive SWOT Analysis")
                swot = st.session_state.generated_swot
                col_s1, col_s2 = st.columns(2)
                
                with col_s1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                                padding: 15px; border-radius: 10px; color: white; margin-bottom: 15px;">
                        <b>✅ STRENGTHS</b><br><small>{swot.get('S', 'N/A')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                                padding: 15px; border-radius: 10px; color: white;">
                        <b>⚠️ WEAKNESSES</b><br><small>{swot.get('W', 'N/A')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_s2:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                padding: 15px; border-radius: 10px; color: white; margin-bottom: 15px;">
                        <b>🚀 OPPORTUNITIES</b><br><small>{swot.get('O', 'N/A')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                padding: 15px; border-radius: 10px; color: white;">
                        <b>⚡ THREATS</b><br><small>{swot.get('T', 'N/A')}</small>
                    </div>
                    """, unsafe_allow_html=True)

# === TAB 3: Q&A CHAT ===
with tab3:
    if not st.session_state.rag_engine:
        st.info("⚠️ Knowledge base not initialized. Run analysis first.")
    else:
        col_h, col_clr = st.columns([6, 1])
        with col_h:
            st.subheader("💬 Context-Aware Q&A")
        with col_clr:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        
        # Chat display
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the video content..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Searching knowledge base..."):
                    try:
                        # RAG retrieval
                        rag_results = st.session_state.rag_engine.search(prompt, n_results=3)
                        context_text = "\n\n".join([res['document'] for res in rag_results])
                        
                        # Generate answer
                        full_prompt = f"""Use the following context from a video transcript to answer the question.
                        
CONTEXT:
{context_text[:2000]}

QUESTION: {prompt}

ANSWER (be concise and factual):"""
                        
                        if st.session_state.orchestrator and st.session_state.orchestrator.summarizer:
                            answer = st.session_state.orchestrator.summarizer.generate(
                                full_prompt,
                                max_new_tokens=300,
                                temperature=0.7
                            )
                        else:
                            answer = "Error: Orchestrator agent not available."
                        
                        message_placeholder.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        with st.expander("📚 View Retrieved Context"):
                            for i, res in enumerate(rag_results):
                                st.caption(f"**Chunk {i+1} (Confidence: {1 - res.get('distance', 1.0):.2f})**")
                                st.text(res.get('document', '')[:300] + "...")
                                
                    except Exception as e:
                        error_msg = f"Error generating answer: {str(e)}"
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

# === TAB 4: PERFORMANCE METRICS ===
with tab4:
    st.markdown("### 📊 System Performance Dashboard")
    
    if not st.session_state.performance_metrics:
        st.info("No metrics available yet. Run an analysis first.")
    else:
        metrics = st.session_state.performance_metrics
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Processing Time", f"{metrics.get('total_time', 0):.1f}s")
        
        with col2:
            retries = sum(metrics.get('retry_counts', {}).values())
            st.metric("Self-Corrections", retries)
        
        with col3:
            cache_stats = metrics.get('cache_stats', {})
            hit_rate = 0
            raw_hit = cache_stats.get('hit_rate', 0)
            if isinstance(raw_hit, str) and '%' in raw_hit:
                try: hit_rate = float(raw_hit.strip('%'))
                except: pass
            else:
                hit_rate = raw_hit * 100 if raw_hit <= 1 else raw_hit
            st.metric("Cache Hit Rate", f"{hit_rate:.1f}%")
        
        with col4:
            q_sum = metrics.get('quality_summary', 'N/A')
            display_q = str(q_sum).split('=')[1].split()[0] if '=' in str(q_sum) else q_sum
            st.metric("Quality Score", display_q)
        
        st.divider()
        
        # Agent Performance Breakdown
        st.subheader("🤖 Agent Latency")
        agent_times = metrics.get('agent_times', {})
        if agent_times:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(agent_times.keys()),
                    y=list(agent_times.values()),
                    marker_color='#667eea',
                    text=[f"{v:.2f}s" for v in agent_times.values()],
                    textposition='auto'
                )
            ])
            fig.update_layout(title="Processing Time per Agent", template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Quality Scores Distribution
        st.subheader("⭐ Output Quality Distribution")
        quality_scores = metrics.get('quality_scores', {})
        if quality_scores:
            score_counts = {}
            for score in quality_scores.values():
                score_name = score.name if hasattr(score, 'name') else str(score)
                score_counts[score_name] = score_counts.get(score_name, 0) + 1
            
            fig3 = go.Figure(data=[
                go.Bar(
                    x=list(score_counts.keys()),
                    y=list(score_counts.values()),
                    marker_color=['#38ef7d', '#00f2fe', '#fee140', '#f5576c'],
                    text=list(score_counts.values()),
                    textposition='auto'
                )
            ])
            fig3.update_layout(title="Quality Ratings Breakdown", template="plotly_dark", height=350)
            st.plotly_chart(fig3, use_container_width=True)

# === FOOTER ===
st.markdown("---")
st.caption("AutoResearcher AI Pro v2.1 | Powered by Enhanced Multi-Agent Intelligence")