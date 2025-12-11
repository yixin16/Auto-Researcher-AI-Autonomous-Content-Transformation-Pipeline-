#  Auto Researcher AI - Self-Correcting Multi-Agent Video Analysis System

<div align="center">


![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)
![MPS](https://img.shields.io/badge/Apple_Silicon-Supported-orange.svg)

**Transform YouTube videos into professional presentations with AI agents that review, correct, and improve each other's work.**

</div>

---

**Auto-Researcher AI** implements a **self-correcting multi-agent system** designed to handle the unpredictability of local LLMs. It features:

- 🛡️ **Robust JSON Engine**: A new parsing layer that handles "chatty" models (like Phi-2) that break standard JSON formatting.
- 🔄 **Automatic Retry Logic**: Regenerates low-quality content with specific improvement feedback from a Critic Agent.
- 🍎 **Cross-Platform**: Native support for NVIDIA GPUs (CUDA) and Apple Silicon (MPS).
- 📊 **Dynamic RAG**: Semantic search that understands context windows across video segments for accurate Q&A.


---


## 🚀 Key Features

### **Intelligence Layer**
- **Self-Correction System**: Agents evaluate their own outputs. If a model hallucinates JSON keys (e.g., outputting is_accurable instead of is_accurate), the system auto-corrects.
- **Critic Agent**: Meta-agent that reviews summaries and insights for hallucination, flow and logical consistency.
- **Fact-Checking Loop**: A dedicated agent compares generated summaries against raw transcripts to flag numerical inconsistencies.

### **Performance**
- **Async Processing**: Analyzes multiple transcript chunks in parallel using asyncio.
- **Smart Caching**: MD5-based cache prevents re-processing known video segments.
- **Resource Management**: Automatically unloads transcription models to free VRAM for reasoning models on consumer hardware.

### **Output Quality**
- **Professional Design**: Generates PowerPoint (.pptx) files with modern gradients, title slides, and smart layouts.
- **Data Visualization**: automatically extracts numerical data from text and renders native charts.
- **SWOT Analysis**: Performs a strategic analysis on the full context of the video.

### **User Experience**
- **Streamlit Dashboard v2.1**:  Real-time metrics, log streaming, and a "Human-in-the-Loop" editor.
- **RAG-Powered Q&A**: Chat with your video content using a vector database (ChromaDB).
- **Deep Metrics**: Track agent latency, self-correction counts, and quality scores per section.

---


##  Installation

### Prerequisites
```bash
Python 3.9+
CUDA-capable GPU (6GB+ VRAM recommended)
yt-dlp (for video downloads)
ffmpeg (for audio processing)
```

### Quick Start

#### 1️⃣ Clone Repository
```bash
git clone https://github.com/yixin16/Auto-Researcher-AI-Autonomous-Content-Transformation-Pipeline-.git
cd Auto-Researcher-AI-Autonomous-Content-Transformation-Pipeline
```

#### 2️⃣ Create Virtual Environment
```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

#### 3️⃣ Install PyTorch with CUDA
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only (not recommended)
pip install torch torchvision torchaudio
```

#### 4️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
```

#### 5️⃣ Environment Configuration
Create `.env` file in project root:
```bash
PEXELS_API_KEY=your_pexels_api_key_here
```

**Get free Pexels API key**: https://www.pexels.com/api/

#### 6️⃣ Create Directory Structure
```bash
mkdir -p outputs/{audio,cache,frames,generated_images,slides}
```

#### 7️⃣ Run Application
```bash
streamlit run app_v2.py
```

Open browser to: `http://localhost:8501`

---

## 🎯 Usage

### Basic Workflow

#### **Step 1: Initialize System**
1. Open sidebar
2. Select model (Phi-2 for speed, Llama-3 for quality)
3. Choose Whisper transcription model
4. Enable/disable features
5. Click "🚀 Initialize System"

#### **Step 2: Analyze Video**
1. Go to "🔍 Analyze" tab
2. Paste YouTube URL
3. Click "▶ Analyze"
4. Watch real-time progress
5. Review quality metrics

#### **Step 3: Review & Edit**
1. Go to "📝 Review & Edit" tab
2. Review AI-generated content
3. Edit summaries, points, insights
4. Check quality scores
5. Click "🎬 Generate Deck"

#### **Step 4: Download & Use**
1. Download PowerPoint file
2. Review SWOT analysis
3. Check performance metrics

#### **Bonus: Interactive Q&A**
1. Go to "💬 Q&A Chat" tab
2. Ask questions about video content
3. Get AI answers based on transcript

---

##  Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   ContentOrchestrator                       │
│  (Manages Context Window, Imports & Error Handling)         │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Robust JSON │    │  RAG Engine  │    │   Utility    │
│    Parser    │    │ (ChromaDB)   │    │   Agents     │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Fact Checker│     │  Context    │     │ Title       │
│ (Strict)    │     │  Classifier │     │ Visual KW   │
│             │     │             │     │ Chart       │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Agent Hierarchy

| Agent | Purpose | Self-Correction | Critic Review |
|-------|---------|----------------|---------------|
| **SummarizerAgent** | Condense transcript sections | ✅ Auto-retry | ✅ Reviewed |
| **KeyPointAgent** | Extract factual takeaways | ✅ Validation | ✅ Reviewed |
| **InsightAgent** | Find strategic implications | ✅ Depth check | ✅ Reviewed |
| **QnAAgent** | Generate discussion questions | ✅ Quality check | ❌ |
| **ChartAgent** | Extract numerical data | ✅ JSON validation | ❌ |
| **SWOTAgent** | Strategic analysis | ✅ Completeness | ❌ |
| **CriticAgent** | Review other agents | N/A | N/A |
| **TitleAgent** | Generate title | ✅ Length check | ❌ |
| **VisualKeywordAgent** | Search terms | ✅ Concreteness | ❌ |

### Processing Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│ 1. INPUT STAGE                                               │
│    YouTube URL → Download Audio → Transcribe (Whisper)      │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. CHUNKING STAGE                                            │
│    Split transcript into 2500-char sections                  │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. PARALLEL ANALYSIS (For each chunk simultaneously)         │
│                                                              │
│    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│    │ Summary Gen  │  │ Points Gen   │  │ Insights Gen │    │
│    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│           │                  │                  │            │
│           ▼                  ▼                  ▼            │
│    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│    │ Self-Eval    │  │ Self-Eval    │  │ Self-Eval    │    │
│    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│           │                  │                  │            │
│           ▼                  ▼                  ▼            │
│    ┌─────────────────────────────────────────────────┐      │
│    │         Critic Agent Review                     │      │
│    │  • Rate quality (1-5)                           │      │
│    │  • Identify issues                              │      │
│    │  • Suggest improvements                         │      │
│    └─────────────────────────────────────────────────┘      │
│                            │                                 │
│           ┌────────────────┴────────────────┐               │
│           ▼                                 ▼               │
│    Quality OK?                        Quality Poor?         │
│    → Continue                         → Retry with feedback │
│                                       → Max 2 retries       │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. RAG INDEXING                                              │
│    Build vector database for Q&A                            │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 5. HUMAN REVIEW                                              │
│    • View quality scores                                     │
│    • Edit content                                            │
│    • Adjust insights                                         │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 6. PRESENTATION GENERATION                                   │
│    • Dynamic layout selection                                │
│    • Visual asset retrieval                                  │
│    • Chart rendering                                         │
│    • SWOT analysis                                           │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 7. OUTPUT                                                    │
│    • Professional PowerPoint                                 │
│    • Performance metrics                                     │
│    • Quality reports                                         │
└──────────────────────────────────────────────────────────────┘
```

---

##  Quality Scoring System


The Critic Agent assigns a score (1-5) to every generation:
Score	Meaning	System Action
5 (Excellent)	Perfect JSON & content	✅ Proceed
3 (Acceptable)	Minor formatting issues	✅ Auto-repair
1 (Poor)	Hallucination detected	🔄 Retry (Max 2x)


### Critic Agent Reviews

#### **Summary Review**
- ✓ Accuracy vs source material
- ✓ Conciseness (3 sentences max)
- ✓ Clarity and readability
- ✓ Completeness

#### **Key Points Review**
- ✓ Relevance (actually important?)
- ✓ Specificity (concrete vs vague)
- ✓ Clarity (understandable?)
- ✓ Actionability

#### **Insights Review**
- ✓ Depth (non-obvious?)
- ✓ Forward-looking perspective
- ✓ Strategic implications
- ✓ Pattern recognition

---


## 🔧 Troubleshooting

### Common Issues

####  **CUDA Out of Memory**

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# 1. Use smaller model
model_choice = "microsoft/phi-2"  # 3GB VRAM
# Instead of: "unsloth/llama-3-8b-Instruct-bnb-4bit"  # 6GB+ VRAM

# 2. Reduce chunk size
def chunk_text(text, max_len=1500):  # Default: 2500

# 3. Disable AI image generation
enable_ai_art = False

# 4. Clear GPU memory
torch.cuda.empty_cache()
```

#### ❌ **Slow Transcription**

**Symptoms**: Whisper takes 5+ minutes

**Solutions**:
```python
# Use smaller Whisper model
whisper_size = "base"    # Fast, acceptable quality
# Instead of: "medium"   # Slower, better quality
# Or: "large"            # Very slow, best quality

# Trade-off:
# base: 2x faster, 5% less accurate
# small: 1.5x faster, 2% less accurate  ⭐ Recommended
# medium: Baseline
# large: 2x slower, 2% more accurate
```

#### ❌ **Low Quality Outputs**

**Symptoms**: Poor summaries, vague points

**Solutions**:
```python
# 1. Enable critic agent
enable_critic = True  # In sidebar

# 2. Increase temperature for creativity
self.generate(prompt, temperature=1.0)  # Default: 0.9

# 3. Use better base model
model_choice = "unsloth/llama-3-8b-Instruct-bnb-4bit"

# 4. Adjust quality thresholds (see Customization section)
```

#### ❌ **Cache Corruption**

**Symptoms**: `TypeError: expected dict, got str`

**Solutions**:
```bash
# Clear cache directory
rm -rf outputs/cache/*

# Or use UI button
# Sidebar → "🗑️ Clear Cache"
```

#### ❌ **yt-dlp Download Fails**

**Symptoms**: `ERROR: Unable to download video`

**Solutions**:
```bash
# Update yt-dlp
pip install -U yt-dlp

# Test download manually
yt-dlp --extract-audio --audio-format m4a "YOUR_URL"

# Check video availability (region locks, age restrictions)
```

#### ❌ **Pexels API Limit**

**Symptoms**: No images in slides

**Solutions**:
```bash
# Check API key in .env
PEXELS_API_KEY=your_key_here

# Free tier: 200 requests/hour
# If exceeded, images will be skipped (not a critical error)

# Or enable AI image generation instead
enable_ai_art = True  # Requires GPU
```

### Core Technologies
- **[Hugging Face Transformers](https://huggingface.co/transformers)**: LLM infrastructure
- **[OpenAI Whisper](https://github.com/openai/whisper)**: Speech-to-text transcription
- **[Microsoft Phi-2](https://huggingface.co/microsoft/phi-2)**: Efficient reasoning model
- **[Meta Llama 3](https://huggingface.co/meta-llama)**: Advanced language understanding
- **[Streamlit](https://streamlit.io)**: Interactive web interface
- **[ChromaDB](https://www.trychroma.com)**: Vector database for RAG

### Libraries & Tools
- **python-pptx**: PowerPoint generation
- **sentence-transformers**: Text embeddings
- **plotly**: Interactive visualizations
- **bitsandbytes**: Model quantization
- **yt-dlp**: Video downloads
- **Pexels API**: Stock photography

### Inspiration
- **LangChain**: Multi-agent frameworks
- **AutoGPT**: Autonomous AI agents
- **BabyAGI**: Task-driven agents
- **Microsoft Semantic Kernel**: Agent orchestration

---
