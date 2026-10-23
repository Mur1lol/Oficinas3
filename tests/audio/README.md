# tests/audio/

This directory holds WAV test fixtures for `test_stt_mock.py`.

## Format requirements

All WAV files must be:
- **Sample rate**: 16,000 Hz (16 kHz)
- **Bit depth**: 16-bit PCM
- **Channels**: Mono (1 channel)

## Generating test files

### Option A — pyttsx3 (offline TTS, no internet)
```bash
pip install pyttsx3
python tests/test_stt_mock.py --generate
```

### Option B — espeak (Raspberry Pi, pre-installed)
```bash
espeak -v en -a 200 -s 130 -w tests/audio/move_e2_e4_raw.wav "move e two e four"
ffmpeg -i tests/audio/move_e2_e4_raw.wav -ar 16000 -ac 1 tests/audio/move_e2_e4.wav
```

### Option C — record your own voice
```bash
arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 tests/audio/move_e2_e4.wav
# Press Ctrl+C after speaking
```

## Expected filenames

| File | Spoken text |
|---|---|
| `move_e2_e4.wav` | "move e two e four" |
| `move_g1_f3.wav` | "move g one f three" |
| `move_a7_a8.wav` | "move a seven a eight" |
| `move_h1_h8.wav` | "move h one h eight" |

These files are excluded from git (add to `.gitignore` if large).
