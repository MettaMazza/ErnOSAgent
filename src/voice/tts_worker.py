"""HIVENET — Kokoro ONNX TTS Worker

Generates speech audio from text using Kokoro ONNX model.
Called as a subprocess from the Rust engine.

Usage:
    python tts_worker.py <text> <output.wav> [--voice am_michael] [--models-dir /path/to/models]
"""
import sys
import os
# Suppress ONNX runtime verbose output to avoid spamming the Rust engine
os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import re
import asyncio
import numpy as np


def sanitize_text(text: str) -> str:
    """Strip unpronounceable characters while preserving punctuation for chunking."""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[\#\*\_\(\)\[\]]', '', text)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0000257F"
        "\U0000FE00-\U0000FE0F"
        "\U0000200D"
        "\U00002600-\U000026FF"
        "\U00002300-\U000023FF"
        "\U00002B50-\U00002B55"
        "\U000023CF-\U000023F3"
        "\U0000203C-\U00003299"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)
    return re.sub(r'\s+', ' ', text).strip()


async def generate():
    if len(sys.argv) < 3:
        print("Usage: python tts_worker.py <text> <output.wav> [--voice am_michael] [--models-dir /path]")
        sys.exit(1)

    text = sys.argv[1]
    output_path = sys.argv[2]

    # Parse optional flags
    voice = "am_michael"
    models_dir = None
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--voice" and i + 1 < len(sys.argv):
            voice = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--models-dir" and i + 1 < len(sys.argv):
            models_dir = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    clean_text = sanitize_text(text)
    if not clean_text:
        print("Text empty after sanitization.")
        sys.exit(0)

    try:
        import soundfile as sf
        from kokoro_onnx import Kokoro

        # Resolve model paths
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")

        model_path = os.path.join(models_dir, "kokoro-v1.0.onnx")
        voices_path = os.path.join(models_dir, "voices-v1.0.bin")

        if not os.path.exists(model_path):
            print(f"ERROR: Model not found at {model_path}")
            sys.exit(1)
        if not os.path.exists(voices_path):
            print(f"ERROR: Voices not found at {voices_path}")
            sys.exit(1)

        kokoro = Kokoro(model_path, voices_path)

        # Chunk by punctuation to simulate natural breaths/pauses
        raw_chunks = re.split(r'([,.!\?]+)', clean_text)

        def split_by_length(t: str, max_len: int = 300):
            words = t.split(' ')
            sub_chunks = []
            curr_chunk = []
            curr_len = 0
            for w in words:
                if curr_len + len(w) + 1 > max_len:
                    if curr_chunk:
                        sub_chunks.append(' '.join(curr_chunk))
                    curr_chunk = [w]
                    curr_len = len(w)
                else:
                    curr_chunk.append(w)
                    curr_len += len(w) + 1
            if curr_chunk:
                sub_chunks.append(' '.join(curr_chunk))
            return sub_chunks

        master_audio = []
        sample_rate = 24000

        for j in range(0, len(raw_chunks), 2):
            text_chunk = raw_chunks[j].strip()
            punct_chunk = raw_chunks[j + 1] if j + 1 < len(raw_chunks) else ""

            if not text_chunk:
                continue

            # Natively govern against Kokoro's 510 token limit
            sub_texts = split_by_length(text_chunk, 300) if len(text_chunk) > 300 else [text_chunk]
            
            for i, sub in enumerate(sub_texts):
                p = punct_chunk if i == len(sub_texts) - 1 else ""
                try:
                    samples, sr = kokoro.create(
                        sub + p,
                        voice=voice,
                        speed=1.0,
                        lang="en-us"
                    )
                    sample_rate = sr
                    if samples is not None and len(samples) > 0:
                        master_audio.append(samples)
                except Exception as e:
                    print(f"WARNING: Chunk generation bypass ({e})")
                    continue

            # Inject silence based on punctuation (only after the final sub-chunk of this logic block)
            if ',' in punct_chunk:
                silence_frames = int(0.25 * sample_rate)
                master_audio.append(np.zeros(silence_frames, dtype=np.float32))
            elif any(p in punct_chunk for p in ['.', '!', '?']):
                silence_frames = int(0.5 * sample_rate)
                master_audio.append(np.zeros(silence_frames, dtype=np.float32))

        if master_audio:
            valid_audio = [arr for arr in master_audio if arr is not None and getattr(arr, "size", 0) > 0]
            if not valid_audio:
                print("ERROR: No valid audio generated after chunking (all lengths zero).")
                sys.exit(1)
                
            final_samples = np.concatenate(valid_audio)
            sf.write(output_path, final_samples, sample_rate)
            print(f"SUCCESS:{output_path}")
        else:
            print("ERROR: No valid audio generated after chunking.")
            sys.exit(1)

    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(generate())
