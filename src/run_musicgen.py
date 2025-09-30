#!/usr/bin/env python3
"""
run_musicgen.py

Run MusicGen inference on input audio files:
1. Preprocess audio (mono, 32kHz)
2. Tokenize audio with EnCodec
3. Feed tokens into MusicGen language model
4. Save logits and tokens to disk

Source notebooks consolidated:
- run_musicgen.ipynb
- z01_feed_audio_into_mgen.ipynb
- z02_get_batch_aud_token*.ipynb
- z03_get_batch_logit*.ipynb

----------------------------------------
Usage:
    python run_musicgen.py --input data/raw/ --output data/processed/logits --model medium

Arguments:
    --input   Path to directory containing audio files (wav, mp3).
    --output  Path to directory where processed logits/tokens will be saved.
    --model   MusicGen model size: [small | medium | large] (default: medium).

Output:
    For each audio file, two numpy files will be created in the output directory:
    - <basename>_tokens.npy   (EnCodec tokens)
    - <basename>_logits.npy   (per-timestep logits from LM)

Example:
    python run_musicgen.py --input ./data/raw/ --output ./data/processed/ --model small
"""

import os
import argparse
import torch
import torchaudio
import numpy as np

from audiocraft.models import MusicGen
from encodec.utils import convert_audio


def preprocess_audio(file_path: str, sample_rate: int = 32000):
    """
    Load audio, convert to mono and resample to 32kHz.
    (based on z01_feed_audio_into_mgen.ipynb)
    """
    wav, sr = torchaudio.load(file_path)
    if wav.shape[0] > 1:  # convert to mono
        wav = wav.mean(dim=0, keepdim=True)
    wav = convert_audio(wav, sr, sample_rate, mono=True)
    return wav, sample_rate


def run_musicgen_on_file(model, file_path: str, output_dir: str):
    """
    Process one audio file with MusicGen:
    - get EnCodec tokens
    - get LM logits
    - save both to disk

    (based on z02_get_batch_aud_token*.ipynb and z03_get_batch_logit*.ipynb)
    """
    os.makedirs(output_dir, exist_ok=True)

    # === Preprocess ===
    wav, sr = preprocess_audio(file_path)

    # === Tokenize & Generate logits ===
    wav = wav.to(model.device)
    tokens = model.compression_model.encode(wav)  # EnCodec tokens
    with torch.no_grad():
        out = model.lm.forward(tokens)  # logits from LM

    logits = out["logits"].cpu().numpy()
    tokens_np = tokens[0].cpu().numpy()

    # === Save ===
    base = os.path.splitext(os.path.basename(file_path))[0]
    np.save(os.path.join(output_dir, f"{base}_tokens.npy"), tokens_np)
    np.save(os.path.join(output_dir, f"{base}_logits.npy"), logits)

    print(f"[INFO] Processed {file_path} → saved tokens & logits to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory containing audio files (wav, mp3)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory to save logits/tokens")
    parser.add_argument("--model", type=str, default="medium",
                        choices=["small", "medium", "large"],
                        help="MusicGen model size (default=medium)")
    args = parser.parse_args()

    # === Load MusicGen (from run_musicgen.ipynb) ===
    print(f"[INFO] Loading MusicGen model ({args.model}) ...")
    model = MusicGen.get_pretrained(args.model)
    model.set_generation_params(duration=20)  # consistent with notebooks
    model.eval()
    model.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(model.device)

    # === Process all files ===
    for fname in os.listdir(args.input):
        if fname.lower().endswith((".wav", ".mp3")):
            run_musicgen_on_file(model,
                                 os.path.join(args.input, fname),
                                 args.output)


if __name__ == "__main__":
    main()
