# Bespoke Speaks 🎙️

A research project exploring natural conversational interactions through a pipeline of Speech-to-Text (STT), Large Language Model (LLM), and Text-to-Speech (TTS) technologies.

## Overview

Bespoke Speaks is an interactive voice chat application that enables natural conversations with an AI assistant. The system uses a three-stage pipeline:

1. **Speech-to-Text**: Converts user speech to text using OpenAI's Whisper model
2. **Language Model**: Processes the text using Qwen2.5 3B, a powerful open-source language model
3. **Text-to-Speech**: Generates natural-sounding responses using CSM (Conversational Speech Model)

## Models Used

- **Whisper** (OpenAI) - For accurate speech recognition
  - [Whisper on Hugging Face](https://huggingface.co/openai/whisper-base)
  - License: MIT

- **Qwen2.5 3B** (Alibaba Cloud) - For natural language understanding and response generation
  - [Qwen2.5 3B on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-3B)
  - License: Apache 2.0

- **CSM** (Sesame) - For high-quality text-to-speech synthesis
  - [CSM on Hugging Face](https://huggingface.co/sesame/csm_1b)
  - License: Apache 2.0

## Quick Start

### Prerequisites

- Python 3.10 or newer
- CUDA-compatible GPU (recommended)
- FFmpeg installed on your system
- Microphone and speakers

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd bespoke-speaks
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Start the voice chat:
```bash
python llmchat.py
```

2. Follow the on-screen prompts:
   - Press Enter to start recording
   - Speak your message
   - Wait for the AI's response
   - Type 'quit' to exit

## Features

- Real-time voice recording with visual feedback
- High-quality speech recognition
- Natural language understanding and response generation
- Natural-sounding voice synthesis
- Error handling and graceful fallbacks
- Visual progress indicators during recording

## Technical Notes

- The application uses Triton for optimization, but compilation is disabled by default for better compatibility
- Audio is recorded at 16kHz sample rate for optimal Whisper performance
- The application automatically uses CUDA if available, falling back to CPU if not

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the Whisper model
- Alibaba Cloud for the Qwen2.5 model
- Sesame for the CSM model
- All contributors and maintainers of the open-source libraries used in this project