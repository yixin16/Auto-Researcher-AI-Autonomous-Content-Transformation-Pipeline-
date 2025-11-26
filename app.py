# app_v2.py - Enhanced Streamlit Interface
import streamlit as st
import os
import time
import torch
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Import Enhanced Orchestrator
from main import ContentOrchestrator

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
    /* Modern Dark Theme */
    .main { background-color: #0E1117; }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
    }
    
    /* Status Indicators */
    .status-excellent { 
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 8px 16px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        display: inline-block;
    }
    
    .status-good {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 8px 16px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        display: inline-block;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 8px 16px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Progress Section */
    .progress-container {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    
    /* Agent Cards */
    .agent-card {
        background-color: #262730;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 3px solid #00d4ff;
    }
    
    .agent-header {
        font-size: 16px;
        font-weight: bold;
        color: #00d4ff;
        margin-bottom: 5px;
    }
    
    .agent-status {
        font-size: 12px;
        color: #aaa;
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

# === SIDEBAR ===
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2040/2040946.png", width=70)
    st.title("⚙️ System Control")
    
    # Hardware Status
    st.subheader("🖥️ Hardware Status")
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            vram_used = torch.cuda.memory_allocated(0) / 1024**3
            vram_free = vram_total - vram_used
            
            st.success(f"**GPU Active:** {gpu_name}")
            st.progress(vram_used / vram_total)
            st.caption(f"VRAM: {vram_used:.1f}GB / {vram_total:.1f}GB used")
        else:
            st.warning("**Running on CPU**")
            st.info("GPU acceleration unavailable")
    except:
        st.error("Hardware check failed")
    
    st.divider()
    
    # Model Configuration
    st.subheader("🤖 AI Configuration")
    
    model_choice = st.selectbox(
        "Reasoning Model",
        ["microsoft/phi-2", "unsloth/llama-3-8b-Instruct-bnb-4bit"],
        help="Phi-2: Fast (3GB VRAM)\nLlama-3: Powerful (6GB+ VRAM)"
    )
    
    whisper_size = st.selectbox(
        "Transcription Model",
        ["base", "small", "medium", "large"],
        index=1,
        help="Larger = More accurate but slower"
    )
    
    st.divider()
    
    # Advanced Features
    st.subheader("🎨 Features")
    
    enable_critic = st.toggle(
        "Quality Control Agent",
        value=True,
        help="Enable self-correction and quality reviews"
    )
    
    enable_ai_art = st.toggle(
        "AI Image Generation",
        value=False,
        help="Generate custom images (requires GPU)"
    )
    
    parallel_processing = st.toggle(
        "Async Processing",
        value=True,
        help="Process chunks in parallel for speed"
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
    else:
        st.success("🟢 **System Active**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("♻️ Restart", use_container_width=True):
                st.session_state.orchestrator = None
                st.session_state.outline = None
                st.session_state.rag_engine = None
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                import shutil
                from pathlib import Path
                cache_dir = Path("outputs/cache")
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                    cache_dir.mkdir()
                st.success("Cache cleared")

# === MAIN CONTENT ===
st.title("🧠 AutoResearcher AI Pro")
st.markdown("### Deep Video Analysis with Self-Correcting Intelligence")

# === TAB LAYOUT ===
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Analyze", 
    "📝 Review & Edit", 
    "💬 Q&A Chat",
    "📊 Performance Metrics"
])

# === TAB 1: ANALYZE ===
with tab1:
    st.markdown("### 🎯 Video Analysis Pipeline")
    
    col_url, col_btn = st.columns([5, 1])
    with col_url:
        url_input = st.text_input(
            "YouTube URL",
            placeholder="https://youtube.com/watch?v=...",
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
            
            # Progress UI
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            logs_placeholder = st.empty()
            
            try:
                with st.spinner("🚀 Launching analysis pipeline..."):
                    import asyncio
                    
                    # Create progress tracking
                    def log_progress(message):
                        st.session_state.processing_logs.append({
                            'time': datetime.now().strftime("%H:%M:%S"),
                            'message': message
                        })
                    
                    # Run analysis
                    start_time = time.time()
                    
                    status_placeholder.markdown('<div class="progress-container">📥 <b>Downloading audio...</b></div>', unsafe_allow_html=True)
                    progress_placeholder.progress(10)
                    
                    status_placeholder.markdown('<div class="progress-container">🎤 <b>Transcribing with Whisper...</b></div>', unsafe_allow_html=True)
                    progress_placeholder.progress(25)
                    
                    status_placeholder.markdown('<div class="progress-container">🧠 <b>Analyzing with AI agents...</b></div>', unsafe_allow_html=True)
                    progress_placeholder.progress(40)
                    
                    # Call orchestrator
                    outline, rag = st.session_state.orchestrator.step_1_analyze(url_input)
                    
                    progress_placeholder.progress(80)
                    status_placeholder.markdown('<div class="progress-container">📚 <b>Building knowledge base...</b></div>', unsafe_allow_html=True)
                    
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
                            <div class="metric-label">Sections Analyzed</div>
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
                            <div class="metric-label">Insights Found</div>
                            <div class="metric-value">{total_insights}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        metrics = st.session_state.performance_metrics
                        retries = sum(metrics['retry_counts'].values())
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Self-Corrections</div>
                            <div class="metric-value">{retries}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.balloons()
                    st.info("👉 **Next Step:** Go to the 'Review & Edit' tab to refine and generate your presentation")
                    
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
        st.caption("Review AI outputs and make adjustments before generating the final deck")
        
        # Quality Overview
        if st.session_state.performance_metrics:
            metrics = st.session_state.performance_metrics
            quality_summary = metrics.get('quality_summary', 'N/A')
            
            st.markdown(f"""
            <div style="background-color: #1e1e1e; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <b>Overall Quality Score:</b> <span class="status-good">{quality_summary}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with st.form("slide_editor_v2"):
            updated_outline = []
            
            for i, slide in enumerate(st.session_state.outline):
                with st.expander(
                    f"📄 Section {slide['id']+1}: {slide['summary'][:60]}...",
                    expanded=(i == 0)
                ):
                    # Quality indicator
                    quality_report = slide.get('quality_report', {})
                    if quality_report:
                        col_q1, col_q2, col_q3 = st.columns(3)
                        with col_q1:
                            st.caption(f"Summary: {quality_report.get('summary_score', 'N/A')}")
                        with col_q2:
                            st.caption(f"Points: {quality_report.get('points_score', 'N/A')}")
                        with col_q3:
                            st.caption(f"Insights: {quality_report.get('insights_score', 'N/A')}")
                    
                    # Content editing
                    c1, c2 = st.columns([3, 1])
                    
                    with c1:
                        new_sum = st.text_area(
                            "Executive Summary",
                            slide['summary'],
                            height=80,
                            key=f"sum_v2_{slide['id']}",
                            help="This appears on the overview slide"
                        )
                    
                    with c2:
                        new_img = st.text_input(
                            "Visual Keyword",
                            slide['image_prompt'],
                            key=f"img_v2_{slide['id']}",
                            help="Used to find stock photos"
                        )
                    
                    c3, c4 = st.columns(2)
                    
                    with c3:
                        points_text = "\n".join(slide['points'])
                        new_pts = st.text_area(
                            "Key Facts (Bullets)",
                            points_text,
                            height=140,
                            key=f"pts_v2_{slide['id']}",
                            help="One point per line"
                        )
                    
                    with c4:
                        insights_text = "\n".join(slide.get('insights', []))
                        new_ins = st.text_area(
                            "Strategic Insights",
                            insights_text,
                            height=140,
                            key=f"ins_v2_{slide['id']}",
                            help="Deep analysis & implications"
                        )
                    
                    updated_outline.append({
                        "id": slide['id'],
                        "summary": new_sum,
                        "points": [p.strip() for p in new_pts.split("\n") if p.strip()],
                        "insights": [i.strip() for i in new_ins.split("\n") if i.strip()],
                        "image_prompt": new_img,
                        "original_text": slide['original_text'],
                        "quality_report": quality_report
                    })
            
            st.divider()
            
            col_gen1, col_gen2, col_gen3 = st.columns([2, 1, 2])
            
            with col_gen2:
                generate_btn = st.form_submit_button(
                    "🎬 Generate Deck",
                    type="primary",
                    use_container_width=True
                )
            
            if generate_btn:
                with st.spinner("🏗️ Building your presentation..."):
                    try:
                        path, title, swot = st.session_state.orchestrator.step_2_generate(
                            updated_outline,
                            st.session_state.current_url,
                            use_ai_images=enable_ai_art
                        )
                        
                        st.success(f"✅ **Presentation Generated:** {title}")
                        
                        # Download button
                        with open(path, "rb") as f:
                            st.download_button(
                                label="📥 Download PowerPoint",
                                data=f,
                                file_name=f"{title}.pptx",
                                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                use_container_width=True
                            )
                        
                        # SWOT Display
                        st.subheader("📊 Executive SWOT Analysis")
                        col_s1, col_s2 = st.columns(2)
                        
                        with col_s1:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                                        padding: 20px; border-radius: 10px; color: white; margin-bottom: 15px;">
                                <h4>✅ STRENGTHS</h4>
                                <p style="font-size: 14px;">{swot.get('S', 'N/A')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                                        padding: 20px; border-radius: 10px; color: white;">
                                <h4>⚠️ WEAKNESSES</h4>
                                <p style="font-size: 14px;">{swot.get('W', 'N/A')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_s2:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                        padding: 20px; border-radius: 10px; color: white; margin-bottom: 15px;">
                                <h4>🚀 OPPORTUNITIES</h4>
                                <p style="font-size: 14px;">{swot.get('O', 'N/A')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                        padding: 20px; border-radius: 10px; color: white;">
                                <h4>⚡ THREATS</h4>
                                <p style="font-size: 14px;">{swot.get('T', 'N/A')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Generation failed: {str(e)}")
                        import traceback
                        with st.expander("Debug Info"):
                            st.code(traceback.format_exc())

# === TAB 3: Q&A CHAT ===
with tab3:
    if not st.session_state.rag_engine:
        st.info("⚠️ Knowledge base not initialized. Run analysis first.")
    else:
        col_h, col_clr = st.columns([6, 1])
        with col_h:
            st.subheader("💬 Interactive Q&A")
        with col_clr:
            if st.button("🗑️ Clear"):
                st.session_state.messages = []
                st.rerun()
        
        # Chat display
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask anything about the video..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    # RAG retrieval
                    docs = st.session_state.rag_engine.search(prompt, n_results=2)
                    context = " ".join(docs) if isinstance(docs, list) else docs
                    
                    # Generate answer
                    full_prompt = f"""Context from video transcript:
{context[:1000]}

Question: {prompt}

Provide a clear, concise answer based on the context above:"""
                    
                    answer = st.session_state.orchestrator.summarizer.generate(
                        full_prompt,
                        max_new_tokens=250,
                        temperature=0.7
                    )
                    
                    st.markdown(answer)
                    st.caption("📚 Answer generated from video transcript")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})

# === TAB 4: PERFORMANCE METRICS ===
with tab4:
    st.markdown("### 📊 System Performance Dashboard")
    
    if not st.session_state.performance_metrics:
        st.info("No metrics available yet. Run an analysis first.")
    else:
        metrics = st.session_state.performance_metrics
        
        # Overview Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Processing Time",
                f"{metrics['total_time']:.1f}s",
                delta=None
            )
        
        with col2:
            retries = sum(metrics['retry_counts'].values())
            st.metric(
                "Self-Corrections",
                retries,
                delta=None,
                help="Number of times agents re-generated content for quality"
            )
        
        with col3:
            cache_stats = metrics.get('cache_stats', {})
            hit_rate = cache_stats.get('hit_rate', 0) * 100
            st.metric(
                "Cache Hit Rate",
                f"{hit_rate:.0f}%",
                delta=None
            )
        
        with col4:
            st.metric(
                "Quality Score",
                metrics.get('quality_summary', 'N/A').split('=')[1].split()[0] if '=' in metrics.get('quality_summary', '') else 'N/A'
            )
        
        st.divider()
        
        # Agent Performance Breakdown
        st.subheader("🤖 Agent Performance")
        
        agent_times = metrics.get('agent_times', {})
        if agent_times:
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=list(agent_times.keys()),
                    y=list(agent_times.values()),
                    marker_color='#667eea',
                    text=[f"{v:.2f}s" for v in agent_times.values()],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Processing Time by Agent",
                xaxis_title="Agent",
                yaxis_title="Time (seconds)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Retry Analysis
        retry_counts = metrics.get('retry_counts', {})
        if retry_counts:
            st.subheader("🔄 Self-Correction Analysis")
            
            col_r1, col_r2 = st.columns([2, 1])
            
            with col_r1:
                fig2 = go.Figure(data=[
                    go.Pie(
                        labels=list(retry_counts.keys()),
                        values=list(retry_counts.values()),
                        hole=0.4
                    )
                ])
                
                fig2.update_layout(
                    title="Retries by Agent",
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            with col_r2:
                st.markdown("##### Retry Breakdown")
                for agent, count in retry_counts.items():
                    st.markdown(f"**{agent}:** {count} corrections")
        
        # Quality Scores Distribution
        st.subheader("⭐ Quality Distribution")
        quality_scores = metrics.get('quality_scores', {})
        
        if quality_scores:
            score_counts = {}
            for score in quality_scores.values():
                score_name = score.name
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
            
            fig3.update_layout(
                title="Quality Scores Across All Outputs",
                xaxis_title="Quality Level",
                yaxis_title="Count",
                template="plotly_dark",
                height=350
            )
            
            st.plotly_chart(fig3, use_container_width=True)

# === FOOTER ===
st.markdown("---")
st.caption("AutoResearcher AI Pro v2.0 | Powered by Enhanced Multi-Agent Intelligence")