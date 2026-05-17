# device_translator.py

import os
import time
import subprocess
import requests
import torch
import sounddevice as sd
import soundfile as sf
import numpy as np

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from faster_whisper import WhisperModel


# =========================================================
# Offline mode
# =========================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# =========================================================
# Folders
# =========================================================
AUDIO_DIR = "audio"
OUTPUT_DIR = "outputs"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# Main model settings
# =========================================================
TRANSLATION_MODEL_NAME = "facebook/m2m100_418M"

SAMPLE_RATE = 16000
CHANNELS = 1

INPUT_WAV = os.path.join(AUDIO_DIR, "device_input.wav")

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# Output cleanup settings
# =========================================================
MAX_OUTPUT_AUDIO_FILES = 4


# =========================================================
# Whisper settings
# =========================================================
WHISPER_MODEL_SIZE = "small"

if torch.cuda.is_available():
    WHISPER_DEVICE = "cuda"
    WHISPER_COMPUTE_TYPE = "float16"
else:
    WHISPER_DEVICE = "cpu"
    WHISPER_COMPUTE_TYPE = "int8"


# =========================================================
# Piper English TTS settings
# =========================================================
PIPER_EXE = "piper/piper.exe"
PIPER_EN_MODEL = "piper/voices/en_US-lessac-medium.onnx"


# =========================================================
# VOICEVOX Japanese TTS settings
# =========================================================
VOICEVOX_URL = "http://127.0.0.1:50021"
VOICEVOX_SPEAKER_ID = 1


# =========================================================
# Load translation model
# =========================================================
print("Loading M2M100 translation model...")

tokenizer = M2M100Tokenizer.from_pretrained(
    TRANSLATION_MODEL_NAME,
    local_files_only=True
)

translation_model = M2M100ForConditionalGeneration.from_pretrained(
    TRANSLATION_MODEL_NAME,
    local_files_only=True
).to(device)

translation_model.eval()


# =========================================================
# Load Whisper model
# =========================================================
print("Loading Whisper model...")

whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE
)

print("\nModels loaded successfully.")
print(f"Translation device: {device}")
print(f"Whisper device: {WHISPER_DEVICE}")
print(f"Whisper model: {WHISPER_MODEL_SIZE}")


# =========================================================
# Clean old output audio files
# =========================================================
def clean_old_audio_files():
    audio_extensions = (".wav", ".mp3", ".m4a")

    files = []

    for filename in os.listdir(OUTPUT_DIR):
        if filename.lower().endswith(audio_extensions):
            path = os.path.join(OUTPUT_DIR, filename)

            if os.path.isfile(path):
                files.append(path)

    files.sort(key=os.path.getmtime, reverse=True)

    old_files = files[MAX_OUTPUT_AUDIO_FILES:]

    for file_path in old_files:
        try:
            os.remove(file_path)
            print("Deleted old audio:", file_path)
        except Exception as e:
            print("Could not delete old audio:", file_path, e)


# =========================================================
# Mode selection
# =========================================================
def choose_mode():
    print("\nChoose translation mode:")
    print("1 = Auto-detect")
    print("2 = Japanese → English")
    print("3 = English → Japanese")

    choice = input("Select mode [1/2/3]: ").strip()

    if choice == "2":
        return "manual_ja_en"

    if choice == "3":
        return "manual_en_ja"

    return "auto"


def decide_direction(detected_language, selected_mode):
    if selected_mode == "manual_ja_en":
        return "ja", "en", "English"

    if selected_mode == "manual_en_ja":
        return "en", "ja", "Japanese"

    lang = (detected_language or "").lower()

    if lang.startswith("ja"):
        return "ja", "en", "English"

    if lang.startswith("en"):
        return "en", "ja", "Japanese"

    return None, None, None


# =========================================================
# Push-to-talk recording
# =========================================================
def record_until_enter(output_path):
    print("\nPress Enter to START recording.")
    input()

    print("Recording... Speak now.")
    print("Press Enter again to STOP recording.")

    chunks = []

    def callback(indata, frames, time_info, status):
        if status:
            print("Audio warning:", status)
        chunks.append(indata.copy())

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=callback
        )

        stream.start()
        input()
        stream.stop()
        stream.close()

    except Exception as e:
        print("Microphone error:", e)
        return False

    if not chunks:
        print("No audio recorded.")
        return False

    audio = np.concatenate(chunks, axis=0).squeeze()

    if audio.size == 0:
        print("Empty audio.")
        return False

    sf.write(output_path, audio, SAMPLE_RATE)

    print("Recording saved:", output_path)
    return True


# =========================================================
# Speech-to-text with Whisper
# =========================================================
def transcribe_audio_auto(audio_path):
    print("Transcribing and detecting language...")

    segments, info = whisper_model.transcribe(
        audio_path,
        vad_filter=True,
        beam_size=5
    )

    text = ""

    for segment in segments:
        text += segment.text

    text = text.strip()
    detected_language = info.language

    print("Detected language:", detected_language)
    print("Recognized text:", text)

    return text, detected_language


# =========================================================
# Translation
# =========================================================
def translate_text(text, src_lang, tgt_lang):
    print(f"Translating {src_lang} → {tgt_lang}...")

    tokenizer.src_lang = src_lang

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = translation_model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
            max_length=128,
            num_beams=5,
            early_stopping=True
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Translation:", result)

    with open("source.txt", "w", encoding="utf-8") as f:
        f.write(text)

    with open("translated.txt", "w", encoding="utf-8") as f:
        f.write(result)

    return result


# =========================================================
# English TTS with Piper
# =========================================================
def create_piper_english_audio(text):
    if not text.strip():
        raise ValueError("No text provided for Piper.")

    if not os.path.exists(PIPER_EXE):
        raise FileNotFoundError(f"Cannot find Piper exe: {PIPER_EXE}")

    if not os.path.exists(PIPER_EN_MODEL):
        raise FileNotFoundError(f"Cannot find Piper model: {PIPER_EN_MODEL}")

    output_path = os.path.join(
        OUTPUT_DIR,
        f"english_output_{int(time.time())}.wav"
    )

    command = [
        PIPER_EXE,
        "--model",
        PIPER_EN_MODEL,
        "--output_file",
        output_path
    ]

    result = subprocess.run(
        command,
        input=text,
        text=True,
        capture_output=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr or "Piper failed.")

    clean_old_audio_files()
    return output_path


# =========================================================
# Japanese TTS with VOICEVOX
# =========================================================
def create_voicevox_japanese_audio(text):
    if not text.strip():
        raise ValueError("No text provided for VOICEVOX.")

    output_path = os.path.join(
        OUTPUT_DIR,
        f"japanese_output_{int(time.time())}.wav"
    )

    query_response = requests.post(
        f"{VOICEVOX_URL}/audio_query",
        params={
            "text": text,
            "speaker": VOICEVOX_SPEAKER_ID
        },
        timeout=30
    )

    query_response.raise_for_status()

    synthesis_response = requests.post(
        f"{VOICEVOX_URL}/synthesis",
        params={
            "speaker": VOICEVOX_SPEAKER_ID
        },
        json=query_response.json(),
        timeout=60
    )

    synthesis_response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(synthesis_response.content)

    clean_old_audio_files()
    return output_path


# =========================================================
# Play audio
# =========================================================
def play_audio(wav_path):
    print("Playing:", wav_path)

    if os.name == "nt":
        subprocess.run(
            ["powershell", "-c", f"Start-Process '{wav_path}'"],
            check=False
        )
    else:
        subprocess.run(
            ["aplay", wav_path],
            check=False
        )


# =========================================================
# One full translation cycle
# =========================================================
def run_once():
    selected_mode = choose_mode()

    success = record_until_enter(INPUT_WAV)

    if not success:
        return

    source_text, detected_language = transcribe_audio_auto(INPUT_WAV)

    if not source_text:
        print("No speech detected. Try again.")
        return

    src_lang, tgt_lang, output_language = decide_direction(
        detected_language,
        selected_mode
    )

    if src_lang is None:
        print("Unsupported or unclear language detected.")
        print("This device currently supports Japanese and English only.")
        return

    translated_text = translate_text(
        source_text,
        src_lang,
        tgt_lang
    )

    if output_language == "English":
        print("Creating English voice with Piper...")
        output_wav = create_piper_english_audio(translated_text)

    elif output_language == "Japanese":
        print("Creating Japanese voice with VOICEVOX...")
        output_wav = create_voicevox_japanese_audio(translated_text)

    else:
        print("Unknown output language.")
        return

    play_audio(output_wav)


# =========================================================
# Main loop
# =========================================================
def main():
    print("\n===================================")
    print(" Offline Japanese ↔ English Device")
    print("===================================")
    print("Modes:")
    print("1 = Auto-detect")
    print("2 = Japanese → English")
    print("3 = English → Japanese")
    print("\nControls:")
    print("Press Enter once to start recording.")
    print("Press Enter again to stop recording.")
    print("Press Ctrl+C to quit.")
    print("\nFor Japanese output, keep VOICEVOX running.")
    print("VOICEVOX check: http://127.0.0.1:50021/version")
    print(f"\nOnly the newest {MAX_OUTPUT_AUDIO_FILES} output audio files will be kept.")

    while True:
        try:
            run_once()

        except requests.exceptions.ConnectionError:
            print("\nCould not connect to VOICEVOX.")
            print("Open VOICEVOX and keep it running.")
            print("Check: http://127.0.0.1:50021/version")

        except KeyboardInterrupt:
            print("\nStopped.")
            break

        except Exception as e:
            print("\nError:", e)


if __name__ == "__main__":
    main()