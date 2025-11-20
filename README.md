# Auto Researcher AI: Autonomous Video Stragegy & Presentation Generator


![alt text](https://img.shields.io/badge/Python-3.10%2B-blue)
![alt text](https://img.shields.io/badge/Frontend-Streamlit-red)
![alt text](https://img.shields.io/badge/AI-Llama3%20%7C%20Whisper%20%7C%20SDXL-purple)
![alt text](https://img.shields.io/badge/License-MIT-green)


AutoResearcher AI Pro is an advanced agentic system that transforms unstructured YouTube video content into professional, strategic PowerPoint presentations.
It utilizes Local LLMs (Llama-3/Phi-2) for reasoning, OpenAI Whisper for transcription, RAG (ChromaDB) for context retrieval, and Stable Diffusion (SDXL Turbo) for generative art—all orchestrated through a user-friendly Streamlit interface.

## Key Features
🕵️ Deep Content Analysis
- **Audio-First Extraction:** Uses yt-dlp and Whisper to transcribe video audio with high accuracy.
- **Multi-Agent Swarm:** Specialized agents for Summarization, Key Points extraction, Strategic Insights, and SWOT Analysis.
- **Data Extraction:** The ChartAgent detects numerical data in the transcript and automatically plots professional bar charts using Matplotlib.

📝 Human-in-the-Loop Workflow
- **Interactive Editor:** Review the AI's generated outline, edit summaries, and refine bullet points before the slides are built
- **Dynamic Slide Engine:** Automatically adapts the presentation layout (3 to 6 slides per section) based on the density of the content.

🎨 Generative Visuals
- **AI Image Generation:** Integrated SDXL Turbo to generate unique, cinematic slide backgrounds based on the context.
- **Stock Photo Fallback:** Optional integration with Pexels API for real-world stock imagery.

💬 RAG Chat ("Chat with Video")
- **Vector Memory:** Indexes the video transcript into ChromaDB.
- **Interactive Q&A:** Ask specific questions about the video (e.g., "What did the speaker say about Q3 earnings?") and get cited answers.

## System Architecture

The system follows a sequential pipeline managed by the `ContentOrchestrator`:

1.  **Audio Downloader (`yt-dlp`, `ffmpeg`):** Downloads the audio track from a YouTube URL and converts it to a usable format.
2.  **Transcriber (`openai-whisper`):** Converts the audio file into a full text transcript.
3.  **Orchestrator:** Manages the workflow and shares a single, memory-efficient LLM instance across all agents.
4.  **LLM Agents (`transformers`, `PyTorch`):** A series of agents process the text in parallel or sequence:
    -   `SummarizerAgent`: Creates a concise summary of the transcript.
    -   `TitleAgent`: Generates a catchy title from the summary.
    -   `KeyPointAgent`: Extracts the most important bullet points.
    -   `QnAAgent`: Creates relevant question-and-answer pairs.
    -   `InsightAgent`: Analyzes the summary to find deeper, non-obvious insights.
5.  **Visual Agent (`Pexels API`):** Searches for a relevant image based on the generated title.
6.  **Presentation Generator (`python-pptx`):** Assembles all the generated content into a polished PowerPoint presentation using a predefined template.

## Tech Stack

-   **Orchestration:** `Python`, `Custom Agent Classes`
-   **UI:** `Streamlit`
-   **LLM:** `Microsoft Phi-2 (CPU/Low VRAM)` , `Unsloth Llama-3 (GPU)`
-   **Vision/Art:** `Diffusers (SDXL Turbo)`
-   **Memory:** `ChromaDB`, `Sentence-Transformers`
-   **Output:** `python-pptx`, `Matplotlib`

## Setup and Installation

**Prerequisites:**
- An NVIDIA GPU with CUDA drivers installed. Verify with `nvidia-smi`.
- FFmpeg installed and accessible in your system's PATH. The easiest way is via Conda.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/yixin16/Auto-Researcher-AI-Autonomous-Content-Transformation-Pipeline-.git
    cd Auto-Researcher-AI-Autonomous-Content-Transformation-Pipeline-
    ```

2.  **Create Conda Environment:**
    ```bash
    conda create --name auto_researcher python=3.11 -y
    conda activate auto_researcher
    ```

3.  **Install PyTorch with CUDA:**
    Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) and get the correct Conda command for your version of CUDA.
    ```bash
    # Example for CUDA 12.1
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

4.  **Install Other Dependencies:**
    ```bash
    pip install -r requirements.txt
    conda install ffmpeg -c conda-forge
    ```
    *(Note: `requirements.txt` should contain libraries like `transformers`, `streamlit`, `openai-whisper`, `python-pptx`, `pexels-api`, etc.)*

## Configuration

1.  **Create `config.yaml`:**
    Create a `config.yaml` file in the root directory to control the system's behavior.
    ```yaml
    # config.yaml
    llm_model: "microsoft/phi-2"
    transcription_model_size: "base" # Options: tiny, base, small, medium, large
    qna_questions_to_generate: 3
    key_points_to_extract: 8
    ```

2.  **Set API Key:**
    Create a file named `.env` in the root directory to store your Pexels API key.
    ```
    PEXELS_API_KEY="YOUR_56_CHARACTER_API_KEY_HERE"
    ```
    *(Note: You'll need to install `pip install python-dotenv` and add `from dotenv import load_dotenv; load_dotenv()` to the top of `agents.py` for this to work automatically.)*

3.  **PowerPoint Template:**
    Place a PowerPoint file named `template.pptx` in the root directory. This will be used as the base for all generated presentations.

## How to Run

1.  **Interactive Web App (Recommended):**
    ```bash
    streamlit run app.py
    ```

2.  **Command-Line Interface:**
    ```bash
    python main.py
    ```

## Future Enhancements
-   **Critic Agent:** Implement a self-correction loop where a "Critic" agent reviews the output of other agents and requests revisions.
-   **Asynchronous Execution:** Convert agent calls to be asynchronous to run non-dependent tasks (like Q&A and Key Points) in parallel.
-   **Advanced Slide Design:** Allow agents to choose different slide layouts based on the content type.