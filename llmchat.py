import os
import torch
import torchaudio
import sounddevice as sd
import numpy as np
import time
import warnings
import importlib.util
from huggingface_hub import hf_hub_download
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
from generator import load_csm_1b, Segment
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

# Filter FFmpeg warning
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Couldn't find ffmpeg or avconv")

# Check if accelerate is available
has_accelerate = importlib.util.find_spec("accelerate") is not None

def install_requirements():
    """Install required packages if missing"""
    if not has_accelerate:
        print("Accelerate package is missing, which is required for efficient model loading.")
        print("Please install it using: pip install 'accelerate>=0.26.0'")
        user_input = input("Do you want to install it now? (y/n): ")
        if user_input.lower() == 'y':
            import subprocess
            subprocess.check_call(["pip", "install", "accelerate>=0.26.0"])
            print("Accelerate installed successfully!")
            return True
    return False

class VoiceChatApp:
    def __init__(self):
        # Select device
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")
        
        # Audio settings
        self.sample_rate = 16000  # Whisper uses 16kHz
        self.recording_duration = 5.0  # seconds (explicitly float)
        
        # Load models
        self.load_models()
        
        # Conversation history
        self.conversation_history = []
    
    def load_models(self):
        print("Loading models...")
        
        # Load Whisper for speech-to-text
        print("Loading Whisper model...")
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(self.device)
        
        # Load text generation model - Qwen2.5 3B
        print("Loading Qwen2.5 3B model...")
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Different loading approaches depending on accelerate availability
        model_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        
        if has_accelerate:
            print("Using accelerate for efficient model loading")
            model_kwargs["device_map"] = "auto"
        else:
            print("Accelerate not found, loading model directly to device")
            # Without accelerate, we don't use device_map
        
        try:
            self.text_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # If accelerate is not available, move model to device manually
            if not has_accelerate:
                self.text_model = self.text_model.to(self.device)
                
        except ImportError as e:
            if "requires Accelerate" in str(e):
                print("Error: accelerate package is required.")
                print("Please install with: pip install 'accelerate>=0.26.0'")
                raise
            else:
                raise
        
        # Load CSM for text-to-speech
        print("Loading CSM model...")
        self.csm_generator = load_csm_1b(self.device)
        
        # Prepare prompts for CSM (from your run_csm.py)
        self.prepare_csm_prompts()
        
        print("All models loaded successfully!")
    
    def prepare_csm_prompts(self):
        # Load default prompts for CSM from HF Hub
        prompt_filepath_conversational_a = hf_hub_download(
            repo_id="sesame/csm-1b",
            filename="prompts/conversational_a.wav"
        )
        
        # Load prompt audio and prepare segment
        audio_tensor = self.load_prompt_audio(prompt_filepath_conversational_a)
        
        prompt_text = (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam I didn't do that "
            "for the exam we had recently to be fair that was a more of a last minute scenario "
            "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
            "sort of start the day with this not like a panic but like a"
        )
        
        self.prompt_segment = Segment(text=prompt_text, speaker=0, audio=audio_tensor)
    
    def load_prompt_audio(self, audio_path: str) -> torch.Tensor:
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        audio_tensor = audio_tensor.squeeze(0)
        # Resample to match CSM sample rate
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=sample_rate, new_freq=self.csm_generator.sample_rate
        )
        return audio_tensor
    
    def record_audio(self) -> np.ndarray:
        """Record audio from the microphone."""
        try:
            print(f"Recording for {self.recording_duration} seconds...")
            print("🎤 ", end="", flush=True)
            
            # Start timer for visual feedback
            start_time = time.time()
            
            # Ensure we use the correct sample rate for Whisper (16kHz)
            audio_data = sd.rec(
                int(self.recording_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,  # Mono audio for Whisper
                dtype=np.float32  # Float32 for better quality
            )
            
            # Visual indicator during recording
            while sd.get_stream().active:
                elapsed = time.time() - start_time
                bars = min(10, int(elapsed / self.recording_duration * 10))
                print("\r🎤 " + "█" * bars + "░" * (10 - bars) + f" {elapsed:.1f}s", end="", flush=True)
                time.sleep(0.1)
            
            # Wait for the recording to complete
            sd.wait()
            print("\rRecording finished ✓" + " " * 20)
            
            # Ensure the audio is properly shaped (flatten if needed)
            audio_data = audio_data.flatten()
            
            # Normalize audio to range [-1, 1]
            if np.abs(audio_data).max() > 0:  # Avoid division by zero
                audio_data = audio_data / np.abs(audio_data).max()
            
            # Check if audio is too quiet (might be silence)
            if np.abs(audio_data).max() < 0.01:
                print("Warning: Audio may be too quiet")
                
            return audio_data
            
        except Exception as e:
            print(f"Error recording audio: {e}")
            # Return silent audio as fallback
            return np.zeros(int(self.recording_duration * self.sample_rate), dtype=np.float32)
    
    def transcribe_audio(self, audio: np.ndarray) -> str:
        """Convert speech to text using Whisper."""
        try:
            # Reshape if needed - Whisper expects a 1D array
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Convert stereo to mono if needed
            
            # Make sure audio is float32 in the range [-1, 1]
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()
            
            # Use the feature extractor directly with proper parameters
            input_features = self.whisper_processor.feature_extractor(
                audio, 
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            ).input_features
            
            # Move to the correct device for inference
            input_features = input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                # Get predicted ids from the model
                predicted_ids = self.whisper_model.generate(
                    input_features,
                    language="en",  # Force English
                    task="transcribe"
                )
                
                # Move back to CPU for decoding
                predicted_ids = predicted_ids.cpu()
            
            # Decode the ids to text
            transcription = self.whisper_processor.tokenizer.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            if not transcription:
                return "I didn't hear anything. Could you please speak again?"
                
            return transcription
            
        except Exception as e:
            print(f"Error in transcription: {e}")
            import traceback
            traceback.print_exc()
            
            return "Sorry, I couldn't transcribe that. Could you try again?"
    
    def create_chat_history(self) -> List[Dict[str, str]]:
        """Convert conversation history to Qwen2 chat format."""
        formatted_history = []
        
        if not self.conversation_history:
            return formatted_history
        
        # Initial system message
        formatted_history.append({
            "role": "system", 
            "content": "You are a helpful AI assistant speaking with a user. Be concise and friendly in your responses."
        })
        
        # Add conversation history
        for message in self.conversation_history:
            if message.startswith("User: "):
                formatted_history.append({
                    "role": "user",
                    "content": message[6:]  # Remove "User: " prefix
                })
            elif message.startswith("AI: "):
                formatted_history.append({
                    "role": "assistant",
                    "content": message[4:]  # Remove "AI: " prefix
                })
        
        return formatted_history
    
    def generate_response(self, user_input: str) -> str:
        """Generate a response using the Qwen2.5 3B model."""
        try:
            # Add user input to history
            self.conversation_history.append(f"User: {user_input}")
            
            # Create chat format for Qwen2.5
            chat_history = self.create_chat_history()
            
            # Format messages for Qwen2.5
            prompt = self.text_tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.text_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                output_ids = self.text_model.generate(
                    inputs.input_ids,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.text_tokenizer.pad_token_id if self.text_tokenizer.pad_token_id else self.text_tokenizer.eos_token_id
                )
            
            # Decode only the newly generated tokens
            response = self.text_tokenizer.decode(
                output_ids[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            # Handle empty responses
            if not response or response.isspace():
                response = "I'm not sure how to respond to that. Could you rephrase your question?"
            
            # Add response to history
            self.conversation_history.append(f"AI: {response}")
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()
            
            # Add fallback response to history
            fallback_response = "I apologize, but I encountered an issue while processing your request. Could we try again?"
            self.conversation_history.append(f"AI: {fallback_response}")
            
            return fallback_response
    
    def speak_response(self, text: str):
        """Convert text to speech using CSM and play it."""
        # Generate audio using CSM
        audio_tensor = self.csm_generator.generate(
            text=text,
            speaker=1,  # Using speaker 1 for the AI
            context=[self.prompt_segment],
            max_audio_length_ms=10_000,
        )
        
        # Always ensure tensor is on CPU before converting to numpy
        audio_np = audio_tensor.cpu().numpy()
        
        # Play audio
        print("Playing response...")
        sd.play(audio_np, self.csm_generator.sample_rate)
        sd.wait()
        print("Playback finished")
    
    def run_conversation_loop(self):
        """Main conversation loop."""
        print("\n=== Voice Chat with Qwen2.5 3B ===")
        print("Press Ctrl+C to exit\n")
        
        # Initial greeting
        print("AI: Hello! I'm your voice assistant. How can I help you today?")
        initial_greeting = "Hello! I'm your voice assistant. How can I help you today?"
        self.conversation_history.append(f"AI: {initial_greeting}")
        self.speak_response(initial_greeting)
        
        try:
            while True:
                # 1. Record user's speech
                print("\n👂 Speak now (or press Ctrl+C to exit)...")
                audio_data = self.record_audio()
                
                # 2. Transcribe speech to text
                print("🔄 Transcribing...")
                user_text = self.transcribe_audio(audio_data)
                print(f"🗣️ You said: \"{user_text}\"")
                
                # Skip processing if the user didn't say anything meaningful
                if user_text in ["Sorry, I couldn't transcribe that. Could you try again?", 
                                "I didn't hear anything. Could you please speak again?"]:
                    print("⚠️ Couldn't hear you clearly. Let's try again.")
                    continue
                
                # 3. Generate AI response
                print("🤔 Generating response...")
                ai_response = self.generate_response(user_text)
                print(f"🤖 AI: \"{ai_response}\"")
                
                # 4. Speak the response
                print("🔊 Speaking response...")
                self.speak_response(ai_response)
                
                # Small delay before next iteration
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n👋 Exiting voice chat. Goodbye!")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    try:
        # Check and install required packages if needed
        requirements_updated = install_requirements()
        if requirements_updated:
            print("Requirements updated. Please restart the application.")
            return
            
        # Initialize and run the app
        app = VoiceChatApp()
        app.run_conversation_loop()
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nIf this is related to missing packages, please install the required packages and try again.")
        print("Common issues:")
        print("- Missing accelerate: pip install 'accelerate>=0.26.0'")
        print("- Missing ffmpeg: sudo apt-get install ffmpeg")

if __name__ == "__main__":
    main() 