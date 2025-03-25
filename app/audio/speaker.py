import torch
import sounddevice as sd
import logging
import threading
from typing import Optional, List, Callable
import time
import numpy as np
import torchaudio
from huggingface_hub import hf_hub_download

from app.utils.event_bus import EventBus
from generator import Segment
from app.audio.csm_service import initialize, generate_speech, cleanup

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
        self._loading_complete = True  # Used for loading indicator
        
        # Register for interruption events
        self.event_bus.subscribe("user_interruption", self._handle_interruption)
        
        # Attempt to load models
        self._load_models()
    
    def _load_models(self) -> bool:
        """
        Initialize the CSM model if not already loaded.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        self._logger.info(f"Loading CSM model on {self.device}...")
        print(f"\n🔊 Loading CSM text-to-speech model...")
        
        # Show loading progress indicator
        loading_thread = threading.Thread(target=self._show_loading_progress)
        loading_thread.daemon = True
        loading_thread.start()
        
        # Flag to track loading status
        self._loading_complete = False
        
        try:
            # Initialize CSM service
            success = initialize(self.device)
            
            # Signal that loading is complete to stop the progress indicator
            self._loading_complete = True
            if loading_thread.is_alive():
                loading_thread.join(timeout=1.0)
            
            if not success:
                self._loading_complete = True
                if loading_thread.is_alive():
                    loading_thread.join(timeout=1.0)
                
                print(f"❌ Failed to load CSM model - speech generation is a critical feature")
                self._logger.error("Failed to load CSM model - speech generation is a critical feature")
                self.event_bus.publish("critical_error", "Text-to-speech system failed. Application cannot function properly without speech capability.")
                return False
                
            self._logger.info("CSM model loaded successfully")
            print(f"✅ CSM model loaded successfully")
            return True
            
        except Exception as e:
            # Signal that loading is complete/failed to stop the progress indicator
            self._loading_complete = True
            if loading_thread.is_alive():
                loading_thread.join(timeout=1.0)
                
            # Print detailed error information
            self._logger.error(f"Error loading CSM model: {e}", exc_info=True)
            print(f"\n❌ Failed to load CSM model: {str(e)}")
            
            # Provide additional troubleshooting information
            if "CUDA out of memory" in str(e):
                print("   Memory error detected. Try closing other applications to free GPU memory.")
            elif "module 'torch' has no attribute 'bfloat16'" in str(e):
                print("   Your PyTorch version might not support bfloat16. Try upgrading PyTorch.")
            elif "ModuleNotFoundError" in str(e):
                print("   Missing module. Ensure you've installed all requirements.")
            
            # This is a critical failure
            self.event_bus.publish("critical_error", "Text-to-speech system failed. Application cannot function properly without speech capability.")
            return False
    
    def _show_loading_progress(self):
        """Show a loading progress indicator while the model is loading."""
        self._loading_complete = False
        indicators = "|/-\\"
        i = 0
        
        while not self._loading_complete:
            print(f"\r📂 Loading CSM model... {indicators[i % len(indicators)]}", end="", flush=True)
            i += 1
            time.sleep(0.2)
    
    def _handle_interruption(self, _: Optional[dict] = None) -> None:
        """Handle user interruption event."""
        self.stop_playback()
        self._logger.info("Playback interrupted by user")
        print("\n🤚 Playback interrupted by user")
    
    def speak(self, 
              text: str, 
              blocking: bool = False) -> None:
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to convert to speech
            blocking: Whether to block until playback is complete
        """
        if not text:
            self._logger.warning("Empty text provided to speak")
            return
        
        # Log but don't print the text - let the caller handle display
        self._logger.info(f"Speaking: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Start playback in a separate thread if non-blocking
        if not blocking:
            self._playback_thread = threading.Thread(
                target=self._speak_impl,
                args=(text,)
            )
            self._playback_thread.daemon = True
            self._playback_thread.start()
        else:
            self._speak_impl(text)

    def _speak_impl(self, text: str) -> None:
        """
        Implementation of the speak method.
        
        Args:
            text: Text to speak
        """
        # Check if we're in cleanup mode and avoid execution
        try:
            import builtins
            if hasattr(builtins, "_bespoke_cleaning_up") and getattr(builtins, "_bespoke_cleaning_up"):
                self._logger.info("Skipping speech generation during application cleanup")
                return
        except:
            pass
            
        # Start playback
        self._is_speaking = True
        self._should_stop_playback = False
        
        # Notify that TTS has started
        self.event_bus.publish("tts_started", {"text": text})
        
        try:
            # Generate speech using simplified approach
            self._logger.info(f"Generating speech for: {text}...")
            # Use logger instead of print to avoid duplicate console messages
            self._logger.info("Generating speech...")
            print(f"🔊 Generating speech... [ID: speech-gen-{id(text)}]")
            
            # Generate audio
            generation_start = time.time()
            
            # No fallback or retry - just generate or fail
            audio_np, sample_rate = generate_speech(text, speaker_id=1)
            
            generation_time = time.time() - generation_start
            self._logger.info(f"Speech generation took {generation_time:.2f} seconds")
            
            # Check if we should stop (user may have interrupted during generation)
            if self._should_stop_playback:
                self._logger.info("Playback canceled before it started")
                return
                
            # Calculate audio duration
            audio_duration = len(audio_np) / sample_rate
            self._logger.info(f"Audio duration: {audio_duration:.2f} seconds")
            
            # Play audio with progress bar
            print(f"🎧 Playing response ({audio_duration:.1f}s)...")
            
            # Start playback
            sd.play(audio_np, sample_rate)
            
            # Show progress during playback
            start_play_time = time.time()
            
            # Monitor playback
            while not self._should_stop_playback and sd.get_stream().active:
                elapsed = time.time() - start_play_time
                progress = min(1.0, elapsed / audio_duration)
                bars = int(progress * 20)
                
                # Don't spam the console with progress updates
                if bars % 5 == 0:
                    print(f"\r🎧 Playing: {bars*5}% [{'▓'*bars}{' '*(20-bars)}]", end="", flush=True)
                    
                time.sleep(0.1)
                
            # If stopped early
            if self._should_stop_playback:
                sd.stop()
                print("\r🎧 Playback stopped                          ")
            else:
                # If completed normally
                print("\r🎧 Playback complete                         ")
                
            # Wait for playback to finish completely (max 0.5s)
            sd.wait(0.5)
            
            # Add a small pause after speech
            time.sleep(0.2)
            
        except Exception as e:
            error_msg = str(e)
            
            # Simple error reporting with no fallback
            if "cache_pos" in error_msg or "AssertionError" in error_msg:
                print("\n❌ Speech generation failed due to CSM KV cache error")
            else:
                print(f"\n❌ Speech generation error: {error_msg}")
            
            self._logger.error(f"Speech generation error: {e}", exc_info=True)
            self.event_bus.publish("speak_error", error_msg)
            
            # Signal critical error - application should exit
            self.event_bus.publish("critical_error", "Text-to-speech system failed. Application cannot function properly without speech capability.")
            
        finally:
            # Always notify that TTS has finished, even if it failed
            self.event_bus.publish("tts_finished", {"text": text})
            
            # Update state
            self._is_speaking = False
            self._should_stop_playback = False
            
            # We don't clean up CSM resources here anymore to preserve the model between calls
            # Only clean up in extreme cases where we need to reset
    
    def stop_playback(self) -> None:
        """Stop audio playback immediately."""
        if self._is_speaking:
            self._should_stop_playback = True
            sd.stop()
            
            # Wait for playback thread to complete
            if self._playback_thread and self._playback_thread.is_alive():
                self._playback_thread.join(timeout=1.0)
                
            self._is_speaking = False
            self._logger.debug("Playback stopped")
    
    def is_speaking(self) -> bool:
        """
        Check if currently speaking.
        
        Returns:
            True if currently speaking, False otherwise
        """
        return self._is_speaking 
        
    def wait_for_playback_complete(self, timeout: float = 30.0) -> None:
        """
        Wait until speech playback is complete.
        
        This method blocks until any ongoing speech playback is complete
        or until the timeout is reached.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        start_time = time.time()
        
        # If we're not speaking, return immediately
        if not self._is_speaking and not self._playback_thread:
            return
            
        # Wait for playback thread to complete
        while (self._is_speaking or 
              (self._playback_thread and self._playback_thread.is_alive())):
            
            # Check if we've exceeded the timeout
            if time.time() - start_time > timeout:
                self._logger.warning(f"Timeout waiting for playback to complete after {timeout}s")
                break
                
            # Short sleep to avoid busy waiting
            time.sleep(0.1)
            
        self._logger.debug("Finished waiting for playback")

    def cleanup(self) -> None:
        """Clean up resources used by the Speaker."""
        try:
            # Stop any ongoing playback
            self.stop_playback()
                
            # Free any CUDA memory used by models
            try:
                cleanup()
            except Exception as e:
                self._logger.error(f"Error during CSM cleanup: {e}")
                
            self._logger.info("Speaker resources cleaned up")
        except Exception as e:
            self._logger.error(f"Error cleaning up Speaker resources: {e}")
            print(f"⚠️ Error cleaning up speaker: {e}") 