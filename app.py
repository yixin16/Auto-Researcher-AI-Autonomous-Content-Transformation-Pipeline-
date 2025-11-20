import streamlit as st
import os
import time
import torch
from dotenv import load_dotenv

# Import the Orchestrator
from main import ContentOrchestrator

# --- 1. CONFIG & PAGE SETUP ---
load_dotenv()
st.set_page_config(
    page_title="AutoResearcher AI Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS (UX ENHANCEMENT) ---
st.markdown("""
<style>
    /* Global Dark Theme Adjustments */
    .reportview-container { background: #0E1117; }
    
    /* Status Badges */
    .gpu-badge { 
        background-color: #00C853; 
        color: white; 
        padding: 4px 8px; 
        border-radius: 4px; 
        font-size: 12px; 
        font-weight: bold; 
        box-shadow: 0 0 10px rgba(0,200,83,0.4);
    }
    .cpu-badge { 
        background-color: #FF6D00; 
        color: white; 
        padding: 4px 8px; 
        border-radius: 4px; 
        font-size: 12px; 
        font-weight: bold; 
    }
    
    /* Result Cards */
    .result-card { 
        background-color: #262730; 
        border: 1px solid #4A4A4A; 
        border-radius: 10px; 
        padding: 20px; 
        margin-bottom: 15px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); 
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #1E1E1E; 
        border-radius: 6px 6px 0 0; 
        color: #B0B0B0; 
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #FF4B4B; 
        color: white; 
    }
    
    /* SWOT Grid */
    .swot-box { padding: 15px; border-radius: 8px; color: #1e1e1e; font-weight: 600; height: 100%; }
</style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if "orchestrator" not in st.session_state: st.session_state.orchestrator = None
if "outline" not in st.session_state: st.session_state.outline = None
if "rag_engine" not in st.session_state: st.session_state.rag_engine = None
if "messages" not in st.session_state: st.session_state.messages = []
if "current_url" not in st.session_state: st.session_state.current_url = ""

# --- 4. SIDEBAR: SETTINGS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2040/2040946.png", width=60)
    st.title("System Config")

    # Hardware Check
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            st.markdown(f'<div class="gpu-badge">⚡ GPU ACTIVE: {gpu_name} ({vram}GB)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="cpu-badge">🐢 RUNNING ON CPU</div>', unsafe_allow_html=True)
    except:
        st.markdown('<div class="cpu-badge">⚠️ HARDWARE CHECK FAILED</div>', unsafe_allow_html=True)

    st.divider()
    
    # Model Selection
    st.subheader("🧠 Intelligence")
    model_choice = st.selectbox(
        "Reasoning Model",
        ["microsoft/phi-2", "unsloth/llama-3-8b-Instruct-bnb-4bit"],
        index=0,
        help="Phi-2 is faster. Llama-3 provides better reasoning but needs 6GB+ VRAM."
    )
    
    whisper_choice = st.selectbox(
        "Transcription Depth",
        ["base", "small", "medium"],
        index=1,
        help="Small is the sweet spot for speed/accuracy."
    )

    # Features
    st.subheader("🎨 Features")
    gen_art_enabled = st.toggle("AI Image Gen (SDXL)", value=False, help="Enable to generate unique art instead of stock photos (Requires GPU).")
    
    st.divider()
    
    # System Init
    if st.session_state.orchestrator is None:
        if st.button("🚀 Initialize System", type="primary", use_container_width=True):
            with st.spinner("Loading Intelligence Engine..."):
                try:
                    config = {
                        'llm_model': model_choice,
                        'transcription_model_size': whisper_choice
                    }
                    st.session_state.orchestrator = ContentOrchestrator(config)
                    st.success("System Online")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Initialization Failed: {e}")
    else:
        st.success("🟢 System Active")
        if st.button("♻️ Reload / Change Model", use_container_width=True):
            st.session_state.orchestrator = None
            st.rerun()

# --- 5. MAIN CONTENT ---
st.title("🧠 AutoResearcher AI Pro")
st.markdown("#### Deep Video Analysis & Strategic Presentation Generator")

# Tab Layout
tab1, tab2, tab3 = st.tabs(["🔍 1. Analyze", "📝 2. Edit & Generate", "💬 3. Chat with Video"])

# --- TAB 1: ANALYZE ---
with tab1:
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        url_input = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed")
    with col_btn:
        analyze_btn = st.button("Start Analysis", type="primary", use_container_width=True)

    if analyze_btn:
        if not st.session_state.orchestrator:
            st.error("⚠️ Please Initialize System in the Sidebar first!")
        elif not url_input:
            st.warning("⚠️ Please enter a valid URL.")
        else:
            st.session_state.current_url = url_input
            
            # UI Progress Tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                with st.status("🕵️ Performing Deep Analysis...", expanded=True) as status:
                    
                    status_text.write("🎧 Downloading Audio Stream...")
                    progress_bar.progress(10)
                    
                    status_text.write("📝 Transcribing & Chunking (Whisper)...")
                    progress_bar.progress(30)
                    
                    # Actual Backend Call
                    outline, rag = st.session_state.orchestrator.step_1_analyze(url_input)
                    
                    status_text.write("🧠 Extracting Insights, Points & Visuals...")
                    progress_bar.progress(80)
                    
                    status_text.write("📚 Indexing Knowledge Base for RAG...")
                    progress_bar.progress(100)
                    
                    st.session_state.outline = outline
                    st.session_state.rag_engine = rag
                    
                    status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                    st.balloons()
                    st.success("Data Ready! Please switch to **Tab 2** to review and generate the deck.")
                    
            except Exception as e:
                st.error(f"Analysis Failed: {e}")

# --- TAB 2: EDIT & GENERATE ---
with tab2:
    if st.session_state.outline is None:
        st.info("👈 Please run an analysis in Tab 1 first.")
    else:
        st.markdown("### ✏️ Human-in-the-Loop Editor")
        st.caption("Review the AI's deep analysis. Add more 'Insights' to trigger dedicated analysis slides.")
        
        with st.form("slide_editor"):
            updated_outline = []
            
            for slide in st.session_state.outline:
                with st.expander(f"Section {slide['id']+1}: {slide['summary'][:50]}...", expanded=False):
                    
                    # Row 1: Summary & Image
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        new_sum = st.text_area("Executive Summary (Overview Slide)", slide['summary'], height=70, key=f"sum_{slide['id']}")
                    with c2:
                        new_img = st.text_input("Image Prompt", slide['image_prompt'], key=f"img_{slide['id']}")
                    
                    # Row 2: Deep Dive (Points & Insights)
                    c3, c4 = st.columns(2)
                    with c3:
                        new_pts = st.text_area("Key Facts (Bullet Points)", "\n".join(slide['points']), height=120, key=f"pts_{slide['id']}")
                    with c4:
                        # Handle missing insights if using old cache
                        current_insights = slide.get('insights', [])
                        new_ins = st.text_area("Strategic Implications (Deep Dive)", "\n".join(current_insights), height=120, key=f"ins_{slide['id']}")
                    
                    updated_outline.append({
                        "id": slide['id'],
                        "summary": new_sum,
                        "points": new_pts.split("\n"),
                        "insights": new_ins.split("\n"),
                        "image_prompt": new_img,
                        "original_text": slide['original_text'] # Pass through for QnA gen
                    })
            
            st.divider()
            
            col_gen_1, col_gen_2 = st.columns([1, 3])
            with col_gen_1:
                generate_btn = st.form_submit_button("🎬 Generate Deck", type="primary", use_container_width=True)
            with col_gen_2:
                st.caption("🚀 **Dynamic Layout Engine:** The system will automatically create 3-6 slides per section based on how much content you added above.")
            
            if generate_btn:
                with st.status("🏗️ Constructing Presentation...", expanded=True) as status:
                    st.write("🎨 Painting Visuals & Fetching Stock Photos...")
                    st.write("📊 Analyzing Data for Charts...")
                    st.write("💡 Generating Q&A pairs...")
                    st.write("📐 Assembling Dynamic Layouts...")
                    
                    try:
                        path, title, swot = st.session_state.orchestrator.step_2_generate(
                            updated_outline, 
                            st.session_state.current_url,
                            use_ai_images=gen_art_enabled
                        )
                        status.update(label="✅ Build Complete!", state="complete", expanded=False)
                        
                        # --- RESULTS DASHBOARD ---
                        st.markdown("### 🎉 Results")
                        
                        # Download Area
                        st.markdown(f"""
                        <div class="result-card">
                            <h3 style="margin:0; color:white;">{title}</h3>
                            <p style="color:#aaa; margin-bottom:10px;">Presentation generated successfully.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with open(path, "rb") as f:
                            st.download_button(
                                label="📥 Download PowerPoint (.pptx)",
                                data=f,
                                file_name=f"{title}.pptx",
                                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                use_container_width=True
                            )
                        
                        # SWOT Grid
                        st.subheader("📊 Executive SWOT")
                        sw1, sw2 = st.columns(2)
                        with sw1:
                            st.markdown(f'<div class="swot-box" style="background:#d4edda; border-left:5px solid #28a745;"><strong>✅ STRENGTHS</strong><br>{swot.get("S")}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="swot-box" style="background:#f8d7da; border-left:5px solid #dc3545;"><strong>🔻 WEAKNESSES</strong><br>{swot.get("W")}</div>', unsafe_allow_html=True)
                        with sw2:
                            st.markdown(f'<div class="swot-box" style="background:#d1ecf1; border-left:5px solid #17a2b8;"><strong>🚀 OPPORTUNITIES</strong><br>{swot.get("O")}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="swot-box" style="background:#fff3cd; border-left:5px solid #ffc107;"><strong>⚠️ THREATS</strong><br>{swot.get("T")}</div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Generation Error: {e}")

# --- TAB 3: CHAT ---
with tab3:
    if st.session_state.rag_engine:
        col_h, col_clr = st.columns([6, 1])
        with col_h:
            st.subheader("💬 Interactive Q&A")
        with col_clr:
            if st.button("Clear"):
                st.session_state.messages = []
                st.rerun()

        # Chat History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat Input
        if prompt := st.chat_input("Ask a specific question about the video content..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            with st.spinner("Consulting Knowledge Base..."):
                # RAG Search
                doc = st.session_state.rag_engine.search(prompt, n_results=1)
                
                # LLM Answer
                full_prompt = f"Use the following video transcript context to answer the question.\n\nContext: '{doc}'\n\nQuestion: {prompt}\n\nAnswer:"
                answer = st.session_state.orchestrator.base.generate(full_prompt, max_new_tokens=300)
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    st.caption("Source Context found in video transcript.")
    else:
        st.info("⚠️ Video is not indexed. Please run analysis in **Tab 1** first.")