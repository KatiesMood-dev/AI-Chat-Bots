#!/usr/bin/env python3
"""
Real-time Whisper transcription from microphone.

- Uses PyAudio to capture audio.
- Buffers 5 seconds of samples (configurable).
- Feeds the buffered data to the small Whisper model.
- Prints the transcribed text on screen.

Author: OpenAI ChatGPT
"""

import os
import sys
import time
import threading
import numpy as np
import pyaudio
import torch
import whisper
import webrtcvad
import queue

# ------------------- Configuration ------------------------------------
CHUNK = 1024                # Number of frames per PyAudio read
FORMAT = pyaudio.paInt16    # 16‑bit int PCM (same as Whisper expects)
CHANNELS = 1                # Mono
RATE = 24000                # 16 kHz – required by Whisper
VAD_RATE = 16000            # VAD requires 16 kHz
WHISPER_MODEL = "large"     # tiny, base, small, medium, large, large-v2, large-v3

SEGMENT_SECONDS = 10        # How many seconds of audio we accumulate before decoding.
DEBUG = True

OUTPUT_FILE = "D:/My Projects/AI-Chat-Bots/Source/STT_Log.txt"

# -----------------------------------------------------------------------

# Buffer to accumulate raw int16 samples
def transcription_thread(buffer):
    # Convert int16 to float32 in [-1.0, 1.0]
    audio_float = buffer.astype(np.float32) / 32768.0
    buffer_cpy = np.copy(buffer)
    buffer = np.array([], dtype=np.int16)

    if DEBUG:
        pa = pyaudio.PyAudio()
        stream_out = pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                frames_per_buffer=CHUNK)
        stream_out.write(frames=buffer_cpy, num_frames=buffer_cpy.shape[0])
        stream_out.close()

    # Whisper expects a numpy array of shape (n_samples,)
    result = model.transcribe(audio_float,
                              fp16=True if torch.cuda.is_available() else False,
                              language="en")  # Change if needed
    with open(OUTPUT_FILE, "a") as f:
        f.write(f"[CONFIDENCE] {result} [TEXT] {result['text']}\n")
    print(f"[{time.strftime('%H:%M:%S')}] {result['text']}")

    # Reset buffer for next segment

# -----------------------------------------------------------------------

def downsample_to_16k(audio_chunk, orig_rate=RATE, target_rate=VAD_RATE):
    """Downsample numpy int16 audio to 16kHz for VAD."""
    if orig_rate == target_rate:
        return audio_chunk
    ratio = orig_rate / target_rate
    indices = np.round(np.arange(0, len(audio_chunk), ratio)).astype(int)
    indices = indices[indices < len(audio_chunk)]
    return audio_chunk[indices]

# -----------------------------------------------------------------------

def vad_worker(vad, audio_queue, buffer_queue):
    """Background thread: consume audio chunks, run VAD, and forward speech chunks."""
    while True:
        audio_chunk = audio_queue.get()
        if audio_chunk is None:
            break

        # Downsample for VAD check
        vad_chunk = downsample_to_16k(audio_chunk)

        # Only check one 30ms frame (fast!)
        frame_length = int(VAD_RATE * 0.03)  # 30 ms
        if len(vad_chunk) >= frame_length:
            frame = vad_chunk[:frame_length].tobytes()
            if vad.is_speech(frame, sample_rate=VAD_RATE):
                buffer_queue.put(audio_chunk)

# -----------------------------------------------------------------------

def main():
    if os.path.isfile(OUTPUT_FILE):
        os.rename(OUTPUT_FILE, f"{OUTPUT_FILE}.{time.strftime('%Y%m%d_%H%M%S')}")

    # Load the model once (takes a few seconds on first run)
    print(f"Loading Whisper '{WHISPER_MODEL}' model…")
    global model
    model = whisper.load_model(WHISPER_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

    # Set up PyAudio
    pa = pyaudio.PyAudio()
    try:
        global stream
        stream = pa.open(format=FORMAT,
                         channels=CHANNELS,
                         rate=RATE,
                         input=True,
                         frames_per_buffer=CHUNK)
    
        
    except Exception as e:
        print(f"Could not open microphone: {e}")
        sys.exit(1)

    # Init WebRTC VAD
    vad = webrtcvad.Vad()
    vad.set_mode(2)

    # Queues for thread comms
    audio_queue = queue.Queue()
    buffer_queue = queue.Queue()

    # Start VAD thread
    threading.Thread(target=vad_worker, args=(vad, audio_queue, buffer_queue), daemon=True).start()

    buffer = np.array([], dtype=np.int16)

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.int16)

            # Push raw audio to VAD worker
            audio_queue.put(audio_chunk)

            # Collect any speech chunks
            while not buffer_queue.empty():
                buffer = np.concatenate((buffer, buffer_queue.get()))

            # Does not mean we transcribe every SEGMENT_SECONDS seconds
            # VAD thread filters out chunks of silence so this waits for SEGMENT_SECONDS seconds of actual speech.
            if len(buffer) >= RATE * SEGMENT_SECONDS:
                threading.Thread(target=transcription_thread,
                                 args=(np.copy(buffer),),
                                 daemon=True).start()
                buffer = np.array([], dtype=np.int16)

        except KeyboardInterrupt:
            break

    # Signal VAD thread to stop
    audio_queue.put(None)


    # Clean up
    stream.stop_stream()
    stream.close()
    pa.terminate()


if __name__ == "__main__":
    main()
