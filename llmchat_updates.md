# Bespoke Speaks: Implementation Plan

This document outlines a comprehensive plan for restructuring and enhancing the Bespoke Speaks application, with a focus on modularity, streaming capabilities, and natural conversation flow.

## Architecture Overview

We will reorganize the application into a modular, event-driven architecture:

```
bespoke-speaks/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Application entry point
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── recorder.py         # Streaming audio recording
│   │   └── speaker.py          # TTS with CSM integration
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── transcriber.py      # Streaming transcription 
│   │   └── llm_service.py      # LLM integration
│   ├── conversation/
│   │   ├── __init__.py
│   │   ├── manager.py          # Conversation state and history
│   │   ├── context.py          # Context management for voice consistency
│   │   └── interruption.py     # Handling speech interruptions
│   └── utils/
│       ├── __init__.py
│       ├── prompt_manager.py   # CSM voice prompt management
│       └── conversation_saver.py # Save conversations to files
└── telephony/  # (Future expansion)
    ├── __init__.py
    └── twilio_service.py       # Twilio integration
```

## Phase 1: Code Restructuring

### 1. Create Modular Package Structure
- Set up directory structure according to architecture plan
- Create empty `__init__.py` files for proper package imports
- Implement core event system for component communication

### 2. Refactor Existing Functionality
- Move audio recording functions from `llmchat.py` to `audio/recorder.py`
- Move transcription code to `nlp/transcriber.py`
- Move LLM interaction to `nlp/llm_service.py`
- Move text-to-speech functionality to `audio/speaker.py`
- Create conversation state management in `conversation/manager.py`

### 3. Create Application Entry Point
- Implement `main.py` with command-line argument parsing
- Set up configuration handling for different operating modes
- Create main application class to orchestrate component interaction

## Phase 2: Feature Enhancements

### 1. Context Management (`conversation/context.py`)
```python
class ConversationContext:
    def __init__(self, initial_prompt_segment):
        self.initial_prompt = initial_prompt_segment
        self.generated_segments = []
        
    def add_segment(self, segment):
        self.generated_segments.append(segment)
        
    def get_context_for_generation(self, max_segments=5):
        # Return a limited window of recent segments to prevent context overflow
        recent_segments = self.generated_segments[-max_segments:] if len(self.generated_segments) > max_segments else self.generated_segments
        return [self.initial_prompt] + recent_segments
```

### 2. Conversation Saving (`utils/conversation_saver.py`)
```python
class ConversationSaver:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        
    def save_audio(self, segments, filename=None):
        """Save segments as a WAV file with timestamps and speaker annotations."""
        if not filename:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"conversation_{timestamp}.wav"
            
        all_audio = torch.cat([seg.audio for seg in segments], dim=0)
        torchaudio.save(
            filename,
            all_audio.unsqueeze(0).cpu(),
            self.sample_rate
        )
        
        # Save metadata alongside audio
        self._save_metadata(segments, filename.replace('.wav', '.json'))
        return filename
```

### 3. Prompt Management (`utils/prompt_manager.py`)
```python
class PromptManager:
    def __init__(self):
        self.prompts = self._load_available_prompts()
        
    def _load_available_prompts(self):
        return {
            "default": {
                "text": "like revising for an exam I'd have to try and like keep up the momentum...",
                "audio_path": "prompts/conversational_a.wav",
                "repo_id": "sesame/csm-1b"
            },
            # Additional prompts can be defined here
        }
        
    def get_prompt_segment(self, prompt_name="default", audio_loader=None):
        """Load and prepare a prompt segment for a specified voice."""
        if prompt_name not in self.prompts:
            prompt_name = "default"
            
        prompt_info = self.prompts[prompt_name]
        audio_path = hf_hub_download(
            repo_id=prompt_info["repo_id"],
            filename=prompt_info["audio_path"]
        )
        
        audio_tensor = audio_loader(audio_path) if audio_loader else self._default_audio_loader(audio_path)
        return Segment(text=prompt_info["text"], speaker=0, audio=audio_tensor)
```

## Phase 3: Streaming Implementation

### 1. Streaming Audio Recorder (`audio/recorder.py`)
```python
class StreamingRecorder:
    def __init__(self, sample_rate=16000, chunk_size=1600):  # 100ms chunks
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.vad = VoiceActivityDetector()  # Implement voice activity detection
        self.event_bus = EventBus()
        
    def start_streaming(self):
        """Begin streaming audio capture with continuous VAD processing."""
        self.buffer = []
        self.is_recording = True
        self.stream = sd.InputStream(
            callback=self._audio_callback,
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.chunk_size
        )
        self.stream.start()
        
    def _audio_callback(self, indata, frames, time, status):
        """Process incoming audio chunks."""
        audio_chunk = indata.copy().flatten()
        self.buffer.append(audio_chunk)
        
        # Check for speech activity
        if self.vad.is_speech(audio_chunk):
            self.event_bus.publish("speech_detected", audio_chunk)
            
        # Check for utterance boundaries
        if self.vad.is_utterance_boundary(self.buffer):
            complete_audio = np.concatenate(self.buffer)
            self.event_bus.publish("utterance_complete", complete_audio)
            self.buffer = []  # Reset buffer after utterance
```

### 2. Streaming Transcriber (`nlp/transcriber.py`)
```python
class StreamingTranscriber:
    def __init__(self, device="cuda"):
        self.device = device
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.transcription_buffer = []
        self.event_bus = EventBus()
        
    def subscribe_to_audio_events(self, event_bus):
        event_bus.subscribe("utterance_complete", self.process_utterance)
        
    def process_utterance(self, audio_data):
        """Process a complete utterance and emit transcription event."""
        input_features = self.whisper_processor.feature_extractor(
            audio_data, 
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device)
        
        with torch.no_grad():
            predicted_ids = self.whisper_model.generate(
                input_features,
                language="en",
                task="transcribe"
            )
            
        transcription = self.whisper_processor.tokenizer.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0].strip()
        
        self.event_bus.publish("transcription_complete", transcription)
```

### 3. Interruption Handling (`conversation/interruption.py`)
```python
class InterruptionHandler:
    def __init__(self, speaker):
        self.speaker = speaker
        self.event_bus = EventBus()
        self.is_speaking = False
        
    def subscribe_to_events(self, event_bus):
        event_bus.subscribe("speech_detected", self.check_for_interruption)
        event_bus.subscribe("tts_started", self.mark_speaking_started)
        event_bus.subscribe("tts_finished", self.mark_speaking_finished)
        
    def check_for_interruption(self, audio_chunk):
        """Check if user is interrupting and should stop TTS."""
        if self.is_speaking and self._is_valid_interruption(audio_chunk):
            self.event_bus.publish("user_interruption", None)
            self.speaker.stop_playback()
            
    def mark_speaking_started(self, _):
        self.is_speaking = True
        
    def mark_speaking_finished(self, _):
        self.is_speaking = False
```

## Immediate Next Steps

1. Create directory structure and initialize git repository with new structure
2. Move core functionality to modules while maintaining existing behavior
3. Implement event bus system for inter-component communication
4. Add context management for improved voice consistency 