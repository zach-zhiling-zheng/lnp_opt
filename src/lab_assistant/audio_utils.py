import os
import wave
import threading
from datetime import datetime
from time import sleep

import pyaudio
import soundfile as sf
import pygame
import keyboard

from openai import OpenAI

def _safe_stop_recording(stream, p):
    try:
        stream.stop_stream()
    except Exception:
        pass
    try:
        stream.close()
    except Exception:
        pass
    try:
        p.terminate()
    except Exception:
        pass

def record_audio(duration=10):
    """
    Record microphone input to both WAV and MP3. Press Space to stop early.
    Returns the MP3 path.
    """
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)

    frames = []
    print("Recording... Press Space to stop.")

    stop_event = threading.Event()

    def listen_for_stop():
        while not stop_event.is_set():
            try:
                if keyboard.is_pressed("space"):
                    stop_event.set()
                    print("Recording stopped by user.")
                    _safe_stop_recording(stream, p)
            except Exception:
                # keyboard might not have permission on some systems
                pass

    listener_thread = threading.Thread(target=listen_for_stop, daemon=True)
    listener_thread.start()

    for _ in range(0, int(fs / chunk * duration)):
        if stop_event.is_set():
            break
        data = stream.read(chunk)
        frames.append(data)

    if not stop_event.is_set():
        _safe_stop_recording(stream, p)

    listener_thread.join()
    print("Recording finished.")

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_filename = f"question_{now}.wav"
    mp3_filename = f"question_{now}.mp3"

    with wave.open(wav_filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b"".join(frames))

    if os.path.getsize(wav_filename) > 0:
        data, samplerate = sf.read(wav_filename)
        sf.write(mp3_filename, data, samplerate, format="MP3")
    else:
        print(f"{wav_filename} is empty.")

    return mp3_filename

def audio_to_text(audio_file_path, api_key):
    client = OpenAI(api_key=api_key)
    with open(audio_file_path, "rb") as audio_file:
        tr = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return tr.text

def text_to_audio(text, output_file_path, api_key):
    client = OpenAI(api_key=api_key)
    with client.audio.speech.with_streaming_response.create(
        model="tts-1", voice="alloy", input=text
    ) as resp:
        resp.stream_to_file(output_file_path)

def play_audio_with_skip(audio_file_path):
    """
    Play audio via pygame with Space to stop. Falls back to direct playback if needed.
    """
    def stop_playback():
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            print("Playback stopped by user.")

    try:
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file_path)
        print("Playing audio... Press Space to skip.")

        stop_event = threading.Event()

        def listen_for_skip():
            while not stop_event.is_set():
                try:
                    if keyboard.is_pressed("space"):
                        stop_event.set()
                        stop_playback()
                        break
                except Exception:
                    # keyboard may fail on some systems
                    pass

        listener_thread = threading.Thread(target=listen_for_skip, daemon=True)
        listener_thread.start()

        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() and not stop_event.is_set():
            sleep(0.1)

        stop_event.set()
        listener_thread.join()
        pygame.mixer.quit()
    except Exception as e:
        print(f"Fallback playback: {e}")
        try:
            import IPython
            from IPython.display import Audio, display
            display(Audio(audio_file_path))
        except Exception as _:
            print("Could not play audio automatically.")
