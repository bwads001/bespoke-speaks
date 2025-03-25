"""
Minimal CSM text-to-speech service using the direct approach from test_csm_minimal.py.
This matches the implementation in llmchat.py which works correctly.
"""

import os
import torch
import torchaudio
import logging
import numpy as np
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

# Configure logging
logger = logging.getLogger("CSMService")

# Global variables - exactly like llmchat.py
_csm_generator = None
_prompt_segment = None
_is_initialized = False

def initialize(device: str = "cuda" if torch.cuda.is_available() else "cpu", force_reinit: bool = False) -> bool:
    """
    Initialize the CSM model using the exact approach from llmchat.py
    
    Args:
        device: Device to use ("cuda" or "cpu")
        force_reinit: Force reinitialization even if already initialized
        
    Returns:
        bool: Success status
    """
    global _csm_generator, _prompt_segment, _is_initialized
    
    # Skip if already initialized and not forcing reinitialization
    if _is_initialized and not force_reinit:
        logger.info("CSM model already initialized, skipping")
        return True
    
    # If we're forcing reinitialization, clean up first
    if force_reinit and _is_initialized:
        cleanup()
    
    try:
        # Load CSM model directly - exactly like llmchat.py and test_csm_minimal.py
        print("Loading CSM model...")
        _csm_generator = load_csm_1b(device)
        print("CSM model loaded successfully!")
        
        # Load default prompts exactly like in llmchat.py
        print("Loading prompt audio...")
        prompt_filepath = hf_hub_download(
            repo_id="sesame/csm-1b",
            filename="prompts/conversational_a.wav"
        )
        
        # Prepare prompt segment - identical to llmchat.py
        audio_tensor, sample_rate = torchaudio.load(prompt_filepath)
        audio_tensor = audio_tensor.squeeze(0)
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=sample_rate, new_freq=_csm_generator.sample_rate
        )
        
        prompt_text = (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam I didn't do that "
            "for the exam we had recently to be fair that was a more of a last minute scenario "
            "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
            "sort of start the day with this not like a panic but like a"
        )
        
        _prompt_segment = Segment(text=prompt_text, speaker=0, audio=audio_tensor)
        _is_initialized = True
        return True
        
    except Exception as e:
        _is_initialized = False
        logger.error(f"Error initializing CSM: {e}", exc_info=True)
        return False

def generate_speech(text: str, speaker_id: int = 1):
    """Generate speech using the exact approach from test_csm_minimal.py"""
    global _csm_generator, _prompt_segment, _is_initialized
    
    if not _is_initialized or _csm_generator is None:
        raise RuntimeError("CSM model not initialized")
        
    if _prompt_segment is None:
        raise RuntimeError("Prompt segment not loaded")
    
    try:
        # Use the exact same parameters from test_csm_minimal.py
        audio_tensor = _csm_generator.generate(
            text=text,
            speaker=speaker_id,
            context=[_prompt_segment],
            max_audio_length_ms=5000,
            temperature=0.7,
            topk=50,
        )
        
        # Convert to numpy
        audio_np = audio_tensor.cpu().numpy()
        return audio_np, _csm_generator.sample_rate
    
    except Exception as e:
        logger.error(f"Error generating speech: {e}", exc_info=True)
        raise

def cleanup():
    """Free resources used by the CSM model"""
    global _csm_generator, _prompt_segment, _is_initialized
    
    # Clean up CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Reset globals
    _csm_generator = None
    _prompt_segment = None
    _is_initialized = False
    
    logger.info("CSM resources cleaned up") 