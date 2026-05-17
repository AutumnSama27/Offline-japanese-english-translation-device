
# Offline Japanese-English Translation Device

This project is an offline Japanese-English audio-to-audio translation device prototype.

The system records speech from a microphone, detects whether the speaker is using Japanese or English, translates the speech into the other language, and outputs the translated result as speech.

## Features

- Offline Japanese ↔ English translation
- Push-to-talk style recording
- Automatic language detection
- Manual fallback mode
- Whisper speech recognition
- M2M100 translation model
- Piper English text-to-speech
- VOICEVOX Japanese text-to-speech
- Automatic cleanup of old output audio files

## System Pipeline

```text
Microphone input
→ Whisper speech recognition
→ Language detection
→ M2M100 translation
→ Piper / VOICEVOX text-to-speech
→ Speaker output
