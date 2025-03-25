import torch
import numpy as np
import logging
from typing import Optional, List
import threading
import time

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from app.utils.event_bus import EventBus

class StreamingTranscriber:
    """
    Speech-to-text transcription using Whisper model.
    
    This class handles speech recognition with streaming support,
    subscribing to audio events and publishing transcription results.
    """
    
    def __init__(self, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 model_name: str = "openai/whisper-base",
                 sample_rate: int = 16000,
                 event_bus: Optional[EventBus] = None):
        """
        Initialize the transcriber.
        
        Args:
            device: Device to run the model on ("cuda" or "cpu")
            model_name: Name of the Whisper model to use
            sample_rate: Audio sample rate (Hz)
            event_bus: Event bus for communication with other components
        """
        self.device = device
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.event_bus = event_bus or EventBus()
        self._logger = logging.getLogger("StreamingTranscriber")
        
        # Lazily loaded models
        self._processor = None
        self._model = None
        
        # Transcription state
        self._is_transcribing = False
        self._transcription_buffer: List[str] = []
        self._transcription_thread: Optional[threading.Thread] = None
    
    def _load_models(self) -> bool:
        """Lazy-load the Whisper model if not already loaded."""
        if self._model is None or self._processor is None:
            self._logger.info(f"Loading Whisper model {self.model_name} on {self.device}...")
            print(f"\n🔄 Loading Whisper model {self.model_name}...")
            try:
                self._processor = WhisperProcessor.from_pretrained(self.model_name)
                self._model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
                self._logger.info("Whisper model loaded successfully")
                print(f"✅ Whisper model loaded successfully!")
                return True
            except Exception as e:
                self._logger.error(f"Error loading Whisper model: {e}")
                print(f"❌ Failed to load Whisper model: {e}")
                return False
        return True
    
    def subscribe_to_audio_events(self) -> None:
        """Subscribe to audio events from the event bus."""
        # Subscribe to both regular and live transcription events
        self.event_bus.subscribe("utterance_complete", self.process_utterance)
        self.event_bus.subscribe("live_transcription_update", self.process_live_transcription)
        
        # Add explicit debug message to show subscription
        print("🔄 Transcriber subscribed to audio events")
        self._logger.info("Transcriber subscribed to utterance_complete and live_transcription_update events")
        
        # Preload the model to avoid delay on first transcription
        self._load_models()
    
    def process_utterance(self, audio_data: np.ndarray) -> None:
        """
        Process a complete utterance asynchronously.
        
        Args:
            audio_data: Audio data as numpy array
        """
        # Print information about the audio data received
        if audio_data is None:
            self._logger.error("Received None audio data in process_utterance")
            print("\n❌ Error: Received empty audio data")
            return
            
        # More detailed logging and clear visual indication
        print("\n" + "+" * 50)
        print("📣 TRANSCRIPTION REQUEST RECEIVED")
        print("+" * 50)
        
        self._logger.info(f"process_utterance called with {len(audio_data)} samples, max amplitude: {np.abs(audio_data).max():.4f}")
        print(f"🔍 Processing audio: {len(audio_data)} samples, duration: {len(audio_data)/self.sample_rate:.2f}s")
        
        # Skip processing if the audio is too short (likely noise)
        if len(audio_data) < 0.2 * self.sample_rate:  # Less than 0.2 seconds (reduced from 0.5)
            self._logger.warning(f"Audio too short ({len(audio_data)/self.sample_rate:.2f}s), skipping transcription")
            print(f"\n⚠️ Audio too short ({len(audio_data)/self.sample_rate:.2f}s), skipping transcription")
            return
            
        # Print the length of the audio data in seconds
        audio_duration = len(audio_data) / self.sample_rate
        self._logger.info(f"Processing utterance of {audio_duration:.2f} seconds")
        print(f"\n🔄 Transcribing audio ({audio_duration:.2f}s)...")
            
        # Start transcription in a separate thread to avoid blocking
        self._transcription_thread = threading.Thread(
            target=self._transcribe_audio,
            args=(audio_data,)
        )
        self._transcription_thread.daemon = True
        self._transcription_thread.start()
        
        # Wait a moment to ensure the thread has started
        time.sleep(0.1)
    
    def process_live_transcription(self, audio_data: np.ndarray) -> None:
        """
        Process ongoing speech for live transcription updates.
        
        Args:
            audio_data: Audio data as numpy array
        """
        if audio_data is None or len(audio_data) < 0.3 * self.sample_rate:  # Require at least 0.3 seconds
            return
            
        # Process in a separate thread to avoid blocking
        live_thread = threading.Thread(
            target=self._transcribe_live_audio,
            args=(audio_data,)
        )
        live_thread.daemon = True
        live_thread.start()
    
    def _transcribe_audio(self, audio: np.ndarray) -> None:
        """
        Convert speech to text using Whisper.
        
        Args:
            audio: Audio data as numpy array
        """
        try:
            self._is_transcribing = True
            self.event_bus.publish("transcription_started", None)
            
            # Load models if not already loaded
            self._load_models()
            
            # Reshape if needed - Whisper expects a 1D array
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Convert stereo to mono if needed
            
            # Make sure audio is float32 in the range [-1, 1]
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()
                
            # Check if audio volume is too low
            max_volume = np.abs(audio).max()
            if max_volume < 0.01:
                self._logger.warning(f"Audio volume very low: {max_volume:.5f}")
                print(f"\n⚠️ Audio volume very low, might not transcribe well")
            
            # Use the feature extractor directly with proper parameters
            input_features = self._processor.feature_extractor(
                audio, 
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            ).input_features
            
            # Move to the correct device for inference
            input_features = input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                # Get predicted ids from the model
                predicted_ids = self._model.generate(
                    input_features,
                    language="en",  # Force English
                    task="transcribe"
                )
                
                # Move back to CPU for decoding
                predicted_ids = predicted_ids.cpu()
            
            # Decode the ids to text
            transcription = self._processor.tokenizer.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            # Handle empty transcriptions
            if not transcription:
                transcription = "I didn't hear anything. Could you please speak again?"
                print("\n❗ Could not transcribe any speech")
            
            # Add to buffer and publish result
            self._transcription_buffer.append(transcription)
            self.event_bus.publish("transcription_complete", transcription)
            
            self._logger.debug(f"Transcription: {transcription}")
            
        except Exception as e:
            self._logger.error(f"Error in transcription: {e}", exc_info=True)
            error_msg = "Sorry, I couldn't transcribe that. Could you try again?"
            print(f"\n❌ Transcription error: {str(e)}")
            self.event_bus.publish("transcription_error", error_msg)
            
        finally:
            self._is_transcribing = False
            self.event_bus.publish("transcription_ended", None)
    
    def _transcribe_live_audio(self, audio: np.ndarray) -> None:
        """
        Transcribe audio for live updates during speech.
        
        Args:
            audio: Audio data as numpy array
        """
        try:
            # Don't set transcribing state - this is a parallel process
            
            # Load models if not already loaded
            if not self._load_models():
                return
                
            # Prepare audio (same as regular transcription)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
                
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
                
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()
                
            # Skip if audio is too quiet
            if np.abs(audio).max() < 0.01:
                return
                
            # Extract features
            input_features = self._processor.feature_extractor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate transcription with faster settings for live updates
            with torch.no_grad():
                predicted_ids = self._model.generate(
                    input_features,
                    language="en",
                    task="transcribe",
                    max_length=128  # Shorter to speed up generation
                )
                
            # Decode
            live_transcription = self._processor.tokenizer.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0].strip()
            
            # Only display if we got something meaningful
            if live_transcription and len(live_transcription) > 1:
                # Simple display - just show the text as it's recognized
                print(f"\r⌨️ {live_transcription}", end="", flush=True)
                
                # Publish event for other components
                self.event_bus.publish("live_transcription_result", live_transcription)
                
        except Exception as e:
            # Log but don't display errors for live transcription
            self._logger.error(f"Error in live transcription: {e}")
    
    def is_transcribing(self) -> bool:
        """
        Check if currently transcribing.
        
        Returns:
            True if currently transcribing, False otherwise
        """
        return self._is_transcribing
    
    def get_transcription_history(self) -> List[str]:
        """
        Get the history of transcriptions.
        
        Returns:
            List of transcribed texts
        """
        return self._transcription_buffer.copy() 