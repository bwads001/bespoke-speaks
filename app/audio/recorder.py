import numpy as np
import sounddevice as sd
import time
import logging
from typing import List, Optional, Callable
import threading

from app.utils.event_bus import EventBus

class VoiceActivityDetector:
    """
    Simple Voice Activity Detector to identify speech in audio.
    
    This class provides methods to detect when someone is speaking
    and when utterances begin or end based on audio energy levels.
    """
    
    def __init__(self, 
                 energy_threshold: float = 0.05,
                 silence_duration: float = 0.7,   # Increased from 0.3 to 0.7 seconds
                 sample_rate: int = 16000):
        """
        Initialize the voice activity detector.
        
        Args:
            energy_threshold: Threshold of audio energy to consider as speech
            silence_duration: Duration of silence (in seconds) to mark end of utterance
            sample_rate: Audio sample rate
        """
        self._energy_threshold = energy_threshold
        self._silence_duration = silence_duration
        self._sample_rate = sample_rate
        self._logger = logging.getLogger("VoiceActivityDetector")
        
        # State tracking
        self._last_speech_time: Optional[float] = None
        self._is_speaking = False
        self._debug_counter = 0
        
        # Keep track of noise floor for adaptive thresholding
        self._noise_levels = []
        self._noise_floor = 0.002  # Initial noise floor estimate
        self._adaptive_threshold = energy_threshold  # Start with the provided threshold
        self._noise_calibration_frames = 0  # Counter for noise calibration
        
        # Counters to ensure reliable detection (avoid false positives)
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        self._min_speech_frames = 3  # Keep speech detection responsive
        self._min_silence_frames = 5  # Increased from 3 to 5 for less aggressive end detection
        
        # Last printed level value
        self._last_printed_level = -1
        
        # Log the initialization parameters
        print(f"🎤 Voice activity detector initialized with energy threshold: {self._energy_threshold}")
        self._logger.info(f"Voice activity detector initialized with energy threshold: {self._energy_threshold}, min_speech_frames: {self._min_speech_frames}, min_silence_frames: {self._min_silence_frames}")
    
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Detect if audio chunk contains speech.
        
        Args:
            audio_chunk: Audio data as numpy array
            
        Returns:
            True if speech is detected, False otherwise
        """
        # Calculate both RMS energy and peak amplitude
        rms_energy = np.sqrt(np.mean(np.square(audio_chunk)))  # True RMS value
        peak_amplitude = np.abs(audio_chunk).max()
        
        # Update noise floor estimate during initial calibration or silence periods
        if self._noise_calibration_frames < 50 or (not self._is_speaking and len(self._noise_levels) < 200):
            self._noise_levels.append(rms_energy)
            self._noise_calibration_frames += 1
            
            if len(self._noise_levels) >= 20:  # Wait until we have enough samples
                # Calculate noise floor as the 10th percentile of observed energy levels
                self._noise_floor = np.percentile(self._noise_levels, 10)
                # Set adaptive threshold higher than the noise floor
                self._adaptive_threshold = max(self._energy_threshold, self._noise_floor * 4)
        
        # Determine if energy is above threshold with hysteresis
        if self._is_speaking:
            # When already speaking, use a lower threshold to continue detection
            is_above_threshold = rms_energy > (self._adaptive_threshold * 0.7)
        else:
            # When not speaking, require a higher threshold to start
            is_above_threshold = rms_energy > self._adaptive_threshold
        
        # Print debug information periodically
        self._debug_counter += 1
        if self._debug_counter % 5 == 0:  # Update more frequently
            self._logger.debug(f"RMS: {rms_energy:.6f}, Peak: {peak_amplitude:.6f}, Threshold: {self._adaptive_threshold:.6f}")
            
            # Create visual audio level meter
            # Check for true silence
            if peak_amplitude < 0.001:  # Lowered true silence threshold
                # Show silent state with full width padding and debug info
                status = "SILENT"
                if self._is_speaking:
                    status += " [SPEAKING]"
                print(f"\r🎤 Level: ▁ ░░░░░░░░░░░░░░░░░░░░   0% | RMS:{rms_energy:.4f} Peak:{peak_amplitude:.4f} TH:{self._adaptive_threshold:.4f} | {status}", end="", flush=True)
                self._last_printed_level = 0
            else:
                # Scale using expected microphone levels
                # Cap at 100% (20 bars) when reaching 0.3
                scaled_value = min(1.0, peak_amplitude / 0.3)  # Cap at 0.3 for 100%
                percentage = int(scaled_value * 100)
                
                # For threshold, use the adaptive threshold relative to max amplitude scale
                threshold_position = int((self._adaptive_threshold / 0.3) * 20)
                threshold_position = min(19, max(0, threshold_position))
                
                # Create visual bar
                bars = int(scaled_value * 20)
                bar = "█" * bars + "░" * (20 - bars)
                
                # Add threshold marker
                if threshold_position < 20:
                    bar_list = list(bar)
                    bar_list[threshold_position] = "▐"  # Threshold marker
                    bar = "".join(bar_list)
                
                # Unicode bar height based on RMS energy
                rms_scaled = min(1.0, rms_energy / 0.3)
                meter = "▁▂▃▄▅▆▇█"[min(7, int(rms_scaled * 8))]
                
                # Add speech state indicator
                status = "ACTIVE" if self._is_speaking else "QUIET"
                if self._consecutive_silence_frames > 0:
                    status += f" [silence:{self._consecutive_silence_frames}/{self._min_silence_frames}]"
                elif self._consecutive_speech_frames > 0:
                    status += f" [speech:{self._consecutive_speech_frames}/{self._min_speech_frames}]"
                
                # Print the audio meter with debug info (only if changed to avoid flickering)
                if percentage != self._last_printed_level:
                    # Clear the entire line before printing new content
                    print(f"\r\033[K🎤 Level: {meter} {bar} {percentage:3d}% | RMS:{rms_energy:.4f} Peak:{peak_amplitude:.4f} TH:{self._adaptive_threshold:.4f} | {status}", end="", flush=True)
                    self._last_printed_level = percentage
        
        # Implement conservative speech detection with hysteresis
        if is_above_threshold:
            self._consecutive_speech_frames += 1
            self._consecutive_silence_frames = 0
        else:
            self._consecutive_speech_frames = 0
            self._consecutive_silence_frames += 1
        
        # Require multiple consecutive frames above threshold to consider it speech
        if self._consecutive_speech_frames >= self._min_speech_frames and not self._is_speaking:
            # Additional check: ensure RMS energy is also high enough
            if rms_energy > self._energy_threshold:
                self._is_speaking = True
                self._last_speech_time = time.time()
                self._logger.info(f"Speech started (RMS: {rms_energy:.6f}, Peak: {peak_amplitude:.4f}, Threshold: {self._adaptive_threshold:.6f})")
                print(f"\n🎙️ Speech detected! [RMS:{rms_energy:.4f} > threshold {self._adaptive_threshold:.4f}]")
                print("🎙️ Speaking...", flush=True)
            else:
                # Reset if RMS too low despite energy threshold
                self._consecutive_speech_frames = 0
        
        # Require consecutive frames below threshold to end speech (hysteresis)
        elif self._is_speaking and self._consecutive_silence_frames >= self._min_silence_frames:
            silence_duration = time.time() - self._last_speech_time
            if silence_duration > self._silence_duration:
                self._is_speaking = False
                self._logger.info(f"Speech ended (silence duration: {silence_duration:.2f}s, RMS: {rms_energy:.6f}, Threshold: {self._adaptive_threshold:.6f})")
                print(f"\n🔍 Speech ended after {silence_duration:.2f}s of silence")
                print("🔍 Processing speech...", flush=True)
                return False  # This will trigger the speech->silence transition
        
        # Update last speech time if we're speaking and energy is above threshold
        if self._is_speaking and is_above_threshold:
            self._last_speech_time = time.time()
        
        return self._is_speaking
    
    def is_utterance_boundary(self, buffer: List[np.ndarray]) -> bool:
        """
        Detect if the current buffer represents an utterance boundary.
        
        Args:
            buffer: List of audio chunks
            
        Returns:
            True if utterance boundary detected, False otherwise
        """
        if not buffer:
            return False
        
        # Detect utterance boundary immediately after stopping speaking
        if not self._is_speaking and self._last_speech_time is not None:
            time_since_speech = time.time() - self._last_speech_time
            if time_since_speech > self._silence_duration * 0.5:  # Reduced to 50% of silence_duration
                self._last_speech_time = None
                self._logger.info(f"Utterance boundary detected after {time_since_speech:.2f}s of silence")
                return True
                
        return False


class StreamingRecorder:
    """
    Streaming audio recorder with voice activity detection.
    
    This class continuously captures audio from the microphone,
    detects speech activity, and publishes events when speech
    is detected or complete utterances are available.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000, 
                 chunk_size: int = 1600, 
                 event_bus: Optional[EventBus] = None):
        """
        Initialize the streaming recorder.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            chunk_size: Size of audio chunks to process
            event_bus: Event bus for publishing events
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.event_bus = event_bus or EventBus()
        self.vad = VoiceActivityDetector(sample_rate=sample_rate)
        self._logger = logging.getLogger("StreamingRecorder")
        
        # State
        self.buffer: List[np.ndarray] = []
        self.is_recording = False
        self.stream: Optional[sd.InputStream] = None
        self._recording_thread: Optional[threading.Thread] = None
        
        # Minimum buffer size before considering it a complete utterance
        self.min_buffer_size = int(0.5 * self.sample_rate / self.chunk_size)  # 0.5 seconds
    
    def start_streaming(self) -> None:
        """Begin streaming audio capture with continuous VAD processing."""
        if self.is_recording:
            self._logger.warning("Recording already in progress")
            return
            
        self.buffer = []
        self.is_recording = True
        
        # Only print the listening message once here, not in _recording_loop
        print("\n👂 Listening for speech...")
        print("┌─────────────────────────────────────┐")
        print("│ 🎙️  Speak now - I'm listening        │")
        print("└─────────────────────────────────────┘")
        
        # Start recording in a separate thread to avoid blocking
        self._recording_thread = threading.Thread(target=self._recording_loop)
        self._recording_thread.daemon = True
        self._recording_thread.start()
        
        self._logger.info("Streaming recorder started")
    
    def stop_streaming(self) -> None:
        """Stop the streaming recorder."""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            
        if self._recording_thread and self._recording_thread.is_alive():
            self._recording_thread.join(timeout=1.0)
            
        self._logger.info("Streaming recorder stopped")
    
    def _recording_loop(self) -> None:
        """Main recording loop that captures audio continuously."""
        try:
            # Get default device name
            try:
                device_info = sd.query_devices(kind='input')
                print(f"\n🎙️ Using microphone: {device_info['name']}")
            except Exception as e:
                self._logger.warning(f"Could not query input device: {e}")
                print("\n🎙️ Using default microphone")
            
            # Initialize stream
            self.stream = sd.InputStream(
                callback=self._audio_callback,
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.chunk_size
            )
            
            # Start streaming
            self.stream.start()
            
            # Add a newline for the audio meter to display on its own line
            # But don't print the initial meter - let the audio callback do it 
            print("\n")
            
            # Keep running until stopped
            while self.is_recording:
                time.sleep(0.1)  # Reduce CPU usage
                
        except Exception as e:
            self._logger.error(f"Error in recording loop: {e}", exc_info=True)
            print(f"\n❌ Recording error: {e}")
            self.event_bus.publish("critical_error", f"Audio recording failed: {e}")
        finally:
            # Clean up
            if hasattr(self, 'stream') and self.stream:
                self.stream.stop()
                self.stream.close()
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """
        Process incoming audio chunks.
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Time info
            status: Status info
        """
        if status:
            self._logger.warning(f"Audio callback status: {status}")
            
        # Use raw audio data without normalization
        audio_chunk = indata.copy().flatten()
        
        # Add to buffer - use raw audio values
        self.buffer.append(audio_chunk)
        
        # Check for speech activity
        was_speaking = self.vad._is_speaking  # Save previous state
        is_speaking = self.vad.is_speech(audio_chunk)
        
        # If speech detected, publish event
        if is_speaking:
            self.event_bus.publish("speech_detected", audio_chunk)
        
        # FORCIBLY check for utterance completion if we stopped speaking
        # This ensures we don't miss the end of speech
        if was_speaking and not is_speaking:
            self._logger.info("Speech->silence transition detected")
            print("\n🔊 Speech->silence transition detected - processing utterance")
            
            # Force a small delay to collect a bit more audio after speech ends
            time.sleep(0.1)
            
            # Clear the level meter line before processing
            print("\r\033[K", end="", flush=True)
            
            # Process the completed utterance if we have enough audio
            # Use a minimum requirement of at least a few frames
            if len(self.buffer) >= 3:  # Very low minimum requirement
                self._process_utterance()
                return
            else:
                print("\n⚠️ Buffer too small, waiting for more audio...")
        
        # Regular check for utterance boundary - make this more sensitive
        if self.vad.is_utterance_boundary(self.buffer) and len(self.buffer) >= 3:
            self._logger.info("Utterance boundary detected via regular check")
            print("\n🔊 Utterance boundary detected via regular check")
            self._process_utterance()
    
    def _process_utterance(self):
        """Process a completed utterance from the buffer."""
        # Combine all audio chunks into a single array
        if not self.buffer:
            self._logger.warning("Empty buffer in _process_utterance")
            return
            
        complete_audio = np.concatenate(self.buffer)
        
        # Check if audio has sufficient energy to be valid speech
        audio_energy = np.mean(np.abs(complete_audio))
        audio_duration = len(complete_audio) / self.sample_rate
        
        # Log the audio energy and duration
        self._logger.info(f"Utterance completed: {audio_duration:.2f}s, energy: {audio_energy:.6f}")
        
        # Very relaxed criteria for valid speech to ensure transcription happens:
        # 1. Minimal duration (at least 0.2 seconds)
        # 2. Any energy above absolute minimum
        valid_energy = audio_energy > self.vad._noise_floor * 2  # Just 2x noise floor
        valid_duration = audio_duration > 0.2  # Even shorter minimum duration
        
        # Print buffer details for debugging
        print(f"\n🎙️ Utterance details: {audio_duration:.2f}s, {len(self.buffer)} chunks, energy: {audio_energy:.6f}")
        
        if valid_energy and valid_duration:
            # Add extra newline to ensure separation from audio level meter
            print("\n")
            self._logger.info(f"Publishing utterance_complete event with {audio_duration:.2f}s of audio (energy: {audio_energy:.6f})")
            print(f"🔄 Processing audio clip: {audio_duration:.1f}s | Energy: {audio_energy:.4f}")
            
            # Publish the speech_ended event
            self.event_bus.publish("speech_ended", complete_audio)
            
            # Publish the utterance_complete event with debug info
            print(f"📢 Sending utterance to transcriber (length: {len(complete_audio)} samples)")
            self.event_bus.publish("utterance_complete", complete_audio)
            
            # Force a small wait to ensure the event has time to be processed
            time.sleep(0.1)
        else:
            reason = "too short" if not valid_duration else "low energy"
            self._logger.info(f"Ignoring audio due to {reason}: duration={audio_duration:.2f}s, energy={audio_energy:.6f}")
            print(f"\n⚠️ Skipping audio ({reason}): duration={audio_duration:.2f}s, energy={audio_energy:.6f}")
        
        # Reset the buffer
        self.buffer = []
    
    def record_for_duration(self, duration: float) -> np.ndarray:
        """
        Record audio for a fixed duration.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Recorded audio as numpy array
        """
        try:
            self._logger.info(f"Recording for {duration} seconds...")
            
            # Calculate frames based on sample rate and duration
            frames = int(duration * self.sample_rate)
            
            # Start timer for visual feedback
            start_time = time.time()
            
            # Record audio
            audio_data = sd.rec(
                frames,
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32
            )
            
            # Visual indicator during recording
            while sd.get_stream().active:
                elapsed = time.time() - start_time
                bars = min(10, int(elapsed / duration * 10))
                progress = "🎤 " + "█" * bars + "░" * (10 - bars) + f" {elapsed:.1f}s"
                print(f"\r{progress}", end="", flush=True)
                time.sleep(0.1)
            
            # Wait for the recording to complete
            sd.wait()
            print("\rRecording finished ✓" + " " * 20)
            
            # Ensure the audio is properly shaped
            audio_data = audio_data.flatten()
            
            # Normalize audio
            if np.abs(audio_data).max() > 0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            # Check if audio is too quiet
            if np.abs(audio_data).max() < 0.01:
                self._logger.warning("Audio may be too quiet")
                
            return audio_data
            
        except Exception as e:
            self._logger.error(f"Error recording audio: {e}")
            # Return silent audio as fallback
            return np.zeros(int(duration * self.sample_rate), dtype=np.float32) 