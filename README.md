# AI Video Analyst & Slide Generator

This project is an advanced multi-agent AI system that automates the process of transforming unstructured video content from YouTube into structured, professional PowerPoint presentations. By leveraging a series of specialized Large Language Model (LLM) agents, the system can download, transcribe, summarize, analyze, and visualize video content with minimal human intervention.

## Key Features

- **Multi-Agent Architecture:** Utilizes an Orchestrator pattern to manage a suite of specialized agents (Summarizer, Key-Point Extractor, Q&A Generator, Title Agent, Visual Agent, etc.) for modular and scalable task execution.
- **End-to-End Automation:** A fully automated pipeline from a YouTube URL to a downloadable `.pptx` file.
- **GPU-Accelerated Performance:** Optimized for NVIDIA GPUs with CUDA, using 4-bit model quantization (`bitsandbytes`) to significantly reduce VRAM usage and improve inference speed.
- **Advanced Content Generation:** Goes beyond summarization to extract non-obvious insights, generate relevant Q&A pairs, and create catchy presentation titles.
- **Dynamic Visuals:** An integrated `VisualAgent` searches for and embeds relevant stock photos based on the presentation's title, enhancing visual appeal.
- **Professional Slide Design:**
    - Uses custom PowerPoint templates (`template.pptx`) for consistent branding and design.
    - Automatically fits text to slide placeholders to prevent overflow.
- **Robust Caching System:** A smart caching layer saves the results of each step (transcription, summarization, etc.), making subsequent runs on the same URL nearly instantaneous.
- **Interactive Web UI:** Includes a user-friendly web interface built with Streamlit for easy access and use.

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

-   **Core:** Python 3.10+
-   **AI/ML:** PyTorch, Hugging Face Transformers, `bitsandbytes`, `openai-whisper`
-   **Data Processing:** `yt-dlp`, `ffmpeg`, `PyYAML`, `requests`
-   **Presentation:** `python-pptx`
-   **Web UI:** Streamlit
-   **Environment:** Conda, NVIDIA CUDA

## Setup and Installation

**Prerequisites:**
- An NVIDIA GPU with CUDA drivers installed. Verify with `nvidia-smi`.
- FFmpeg installed and accessible in your system's PATH. The easiest way is via Conda.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/Auto_Researcher_AI.git
    cd Auto_Researcher_AI
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