import os
import torch
import torchaudio
import sounddevice as sd
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

def main():
    # Select device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Load CSM model exactly as in llmchat.py
    print("Loading CSM model...")
    generator = load_csm_1b(device)
    print("CSM model loaded successfully!")
    
    # Load the prompt audio
    print("Loading prompt audio...")
    prompt_filepath = hf_hub_download(
        repo_id="sesame/csm-1b",
        filename="prompts/conversational_a.wav"
    )
    
    # Prepare prompt segment
    audio_tensor, sample_rate = torchaudio.load(prompt_filepath)
    audio_tensor = audio_tensor.squeeze(0)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    
    prompt_text = (
        "like revising for an exam I'd have to try and like keep up the momentum because I'd "
        "start really early I'd be like okay I'm gonna start revising now and then like "
        "you're revising for ages and then I just like start losing steam I didn't do that "
        "for the exam we had recently to be fair that was a more of a last minute scenario "
        "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
        "sort of start the day with this not like a panic but like a"
    )
    
    prompt_segment = Segment(text=prompt_text, speaker=0, audio=audio_tensor)
    
    # Generate a simple greeting
    print("Generating speech for greeting...")
    greeting = "Hello. This is a short test."
    
    # Use exact parameters from llmchat.py
    audio_tensor = generator.generate(
        text=greeting,
        speaker=1,
        context=[prompt_segment],
        max_audio_length_ms=5000,
        temperature=0.7,
        topk=50,
    )
    
    print("Speech generated successfully!")
    
    # Play the audio
    print(f"Playing audio...")
    audio_np = audio_tensor.cpu().numpy()
    sd.play(audio_np, generator.sample_rate)
    sd.wait()
    print("Playback finished")
    
    # Clean up
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main() 