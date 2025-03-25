import torch
import sounddevice as sd
import logging
import threading
from typing import Optional, List, Callable
import time
import numpy as np
import torchaudio
from huggingface_hub import hf_hub_download
import asyncio

from app.utils.event_bus import EventBus
from app.utils.model_loader import ModelLoader
from app.models.generator import get_or_create_generator, prepare_prompt_segment, Segment

class Speaker:
    """
    Text-to-speech component using CSM model.
    
    This class handles text-to-speech conversion using the CSM model
    and manages audio playback with support for interruptions.
    """
    
    def __init__(self, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 event_bus: Optional[EventBus] = None):
        """
        Initialize the speaker.
        
        Args:
            device: Device to run the model on ("cuda" or "cpu")
            event_bus: Event bus for communication with other components
        """
        self.device = device
        self.event_bus = event_bus or EventBus()
        self._logger = logging.getLogger("Speaker")
        
        # Playback state
        self._is_speaking = False
        self._playback_thread: Optional[threading.Thread] = None
        self._should_stop_playback = False
        
        # Add a lock to prevent concurrent generation
        self._generation_lock = threading.Lock()
        self._is_generating = False
        
        # Register for interruption events
        self.event_bus.subscribe("user_interruption", self._handle_interruption)
        
        # Register to provide generation status
        self.event_bus.subscribe("is_speech_generating", self._handle_is_generating_query)
        
        # Attempt to load models
        self._load_models()
    
    def _load_models(self) -> bool:
        """
        Initialize the CSM model if not already loaded.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        self._logger.info(f"Loading CSM model on {self.device}...")
        
        try:
            # Use the model loader utility with a custom load function
            def load_csm_components():
                # Get the generator and prompt segment
                generator = get_or_create_generator(self.device)
                prompt_segment = prepare_prompt_segment(self.device)
                
                return {
                    "generator": generator,
                    "prompt_segment": prompt_segment
                }
            
            # Use the unified model loader
            result = ModelLoader.load_model(
                load_function=load_csm_components,
                model_name="CSM 1B",
                device=self.device,
                model_type="speech",
                event_bus=self.event_bus
            )
            
            if not result:
                self._logger.error("Failed to load CSM model - speech generation is a critical feature")
                self.event_bus.publish("critical_error", "Text-to-speech system failed. Application cannot function properly without speech capability.")
                return False
            
            # Store references to the loaded components
            self.csm_generator = result["generator"]
            self.prompt_segment = result["prompt_segment"]
            
            self._logger.info("CSM model loaded successfully")
            return True
            
        except Exception as e:
            # Print detailed error information
            self._logger.error(f"Error loading CSM model: {e}", exc_info=True)
            
            # This is a critical failure
            self.event_bus.publish("critical_error", "Text-to-speech system failed. Application cannot function properly without speech capability.")
            return False
    
    def _handle_interruption(self, _: Optional[dict] = None) -> None:
        """Handle user interruption event."""
        self.stop_playback()
        self._logger.info("Playback interrupted by user")
        print("\n🤚 Playback interrupted by user")
    
    def _handle_is_generating_query(self, _: Optional[dict] = None) -> bool:
        """
        Handle query about generation status.
        
        Returns:
            bool: True if currently generating speech, False otherwise
        """
        return self._is_generating
    
    def speak(self, text: str, voice_id: int = 0) -> bool:
        """
        Generate and play speech for the given text.
        
        Args:
            text: The text to speak
            voice_id: The voice ID to use
            
        Returns:
            True if generation was successful, False otherwise
        """
        if self._is_generating:
            self._logger.warning("Speech generation already in progress, ignoring request")
            return False
        
        try:
            # Use threading instead of asyncio
            self._playback_thread = threading.Thread(
                target=self._speak_impl,
                args=(text, voice_id)
            )
            self._playback_thread.daemon = True
            self._playback_thread.start()
            return True
        except Exception as e:
            self._logger.error(f"Error speaking text: {e}")
            self._is_generating = False
            return False

    def _speak_impl(self, text: str, voice_id: int) -> bool:
        """Implementation of speech generation and playback"""
        generation_start = time.time()
        
        with self._generation_lock:
            if self._is_generating:
                return False
            
            self._is_generating = True
            self.event_bus.publish("speech_generation_started", None)
            
            try:
                # Generate speech
                self._logger.info(f"Generating speech for text: {text}")
                print(f"🔊 Generating speech...")
                
                # Get context segments for voice consistency
                context_segments = self.event_bus.publish_and_get_result(
                    "get_tts_context", 
                    None, 
                    default_result=[self.prompt_segment]
                )
                
                self._logger.info(f"Using {len(context_segments)} context segments for voice consistency")
                
                # Use simple sequential generation to ensure reliability
                audio = self.csm_generator.generate(
                    text=text,
                    speaker=voice_id,
                    context=context_segments,
                    batch_size=1,  # Use single token generation for reliability
                    temperature=0.7,
                    topk=50,
                    use_kv_caching=True
                )
                
                generation_time = time.time() - generation_start
                self._logger.info(f"Speech generation completed in {generation_time:.2f}s")
                
                # Create a segment for the generated speech
                segment = Segment(text=text, speaker=voice_id, audio=audio)
                
                # Publish the segment for context tracking
                self.event_bus.publish("speech_generated", segment)
                
                # Play the generated audio
                self._play_audio(audio)
                
                return True
            except Exception as e:
                self._logger.error(f"Error in speech generation: {e}")
                print(f"\n❌ Speech generation error: {str(e)}")
                return False
            finally:
                self._is_generating = False
                self.event_bus.publish("speech_generation_completed", None)

    def _play_audio(self, audio: torch.Tensor) -> None:
        """
        Play the generated audio.
        
        Args:
            audio: The generated audio tensor
        """
        try:
            # Convert to numpy for playback
            audio_np = audio.cpu().numpy()
            
            # Calculate audio duration
            sample_rate = self.csm_generator.sample_rate
            audio_duration = len(audio_np) / sample_rate
            self._logger.info(f"Playing audio ({audio_duration:.2f}s)")
            
            # Temporary buffer to allow playback adjustments
            # (e.g., volume control could be added here in the future)
            self._audio_buffer = audio_np
            
            # Start playback
            self._is_speaking = True
            self._should_stop_playback = False
            self.event_bus.publish("playback_started", {"duration": audio_duration})
            
            # Play the audio
            start_play_time = time.time()
            sd.play(audio_np, sample_rate)
            
            # Monitor playback
            while not self._should_stop_playback and sd.get_stream().active:
                elapsed = time.time() - start_play_time
                progress = min(1.0, elapsed / audio_duration)
                
                # Update progress (optional)
                self.event_bus.publish("playback_progress", {
                    "elapsed": elapsed,
                    "total": audio_duration,
                    "progress": progress
                })
                
                # Yield to other threads
                time.sleep(0.1)
            
            # If stopped early
            if self._should_stop_playback:
                sd.stop()
                print("\r🎧 Playback stopped                          ")
            else:
                # If completed normally
                print("\r🎧 Playback complete                         ")
                
            # Reset speaking flag
            self._is_speaking = False
            
            # Signal playback completion
            self.event_bus.publish("playback_finished", None)
            
            # Add a small pause after speech
            time.sleep(0.2)
        except Exception as e:
            self._logger.error(f"Error during audio playback: {e}")
            self._is_speaking = False
            self.event_bus.publish("playback_error", str(e))

    def stop_playback(self) -> None:
        """Stop any ongoing speech playback."""
        self._should_stop_playback = True
        
        # Stop audio playback
        sd.stop()
        
        self._logger.info("Playback stopped")
    
    def is_speaking(self) -> bool:
        """
        Check if currently speaking.
        
        Returns:
            bool: True if currently speaking, False otherwise
        """
        return self._is_speaking
        
    def is_generating(self) -> bool:
        """
        Check if currently generating speech.
        
        Returns:
            bool: True if currently generating speech, False otherwise
        """
        return self._is_generating

    def wait_for_playback_complete(self, timeout: float = 60.0) -> bool:
        """
        Wait for audio playback to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if playback completed successfully, False if timed out
        """
        start_time = time.time()
        print(f"⏳ Waiting for speech generation and playback to complete...")
        
        # First wait for generation to complete
        while self._is_generating:
            if time.time() - start_time > timeout:
                self._logger.warning(f"Timed out waiting for speech generation after {timeout}s")
                print(f"⚠️ Timed out waiting for speech generation")
                return False
            time.sleep(0.1)
        
        # Then wait for playback to complete
        while self._is_speaking:
            if time.time() - start_time > timeout:
                self._logger.warning(f"Timed out waiting for speech playback after {timeout}s")
                print(f"⚠️ Timed out waiting for speech playback")
                return False
            time.sleep(0.1)
        
        elapsed = time.time() - start_time
        if elapsed > 1.0:  # Only log if we actually had to wait
            print(f"✅ Speech generation and playback completed in {elapsed:.1f}s")
        
        return True

    def cleanup(self) -> None:
        """Clean up resources used by the speaker."""
        # Stop any ongoing playback
        self.stop_playback()
        
        # Wait for any ongoing speech to complete
        self.wait_for_playback_complete(timeout=2.0)
        
        # No explicit resource cleanup needed for CSM model
        # It will be garbage collected when the app exits
        self._logger.info("Speaker resources cleaned up") 