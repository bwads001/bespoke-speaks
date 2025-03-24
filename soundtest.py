import sounddevice as sd
import numpy as np
import sys

# Print available devices first
print("Available audio devices:")
devices = sd.query_devices()
print(devices)

# Find default devices
try:
    default_input = sd.query_devices(kind='input')
    default_output = sd.query_devices(kind='output')
    print(f"\nDefault input device: {default_input['name']}")
    print(f"Default output device: {default_output['name']}")
except Exception as e:
    print(f"Error getting default devices: {e}")

# Check if any devices are available
if len(devices) == 0:
    print("No audio devices found. Please check your audio configuration.")
    sys.exit(1)

# Test recording
duration = 3  # seconds
fs = 44100  # Sample rate

try:
    print(f"\nRecording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Recording finished")

    # Test playback
    print("Playing back recording...")
    sd.play(recording, fs)
    sd.wait()
    print("Playback finished")
except Exception as e:
    print(f"Error during audio operation: {e}")
    print("\nTroubleshooting tips:")
    print("1. Make sure PulseAudio is running: 'pulseaudio --start'")
    print("2. Check if your WSL environment has access to Windows audio devices")
    print("3. Try setting a specific device with 'sd.default.device = [device_id]'")
    print("4. Verify PULSE_SERVER environment variable is set correctly")