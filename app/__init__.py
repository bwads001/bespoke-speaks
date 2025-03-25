"""
Bespoke Speaks - Natural Voice Conversation System
"""

__version__ = "0.1.0"

# Make key components available at package level
from app.main import BespokeApp, main
from app.utils.event_bus import EventBus, global_event_bus

# Silence warnings during import
import warnings
import os

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

# Filter FFmpeg warning
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Couldn't find ffmpeg or avconv")
