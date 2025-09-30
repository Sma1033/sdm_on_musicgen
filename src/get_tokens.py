#!/usr/bin/env python3
"""
get_tokens.py

Convert audio files into EnCodec tokens using MusicGen's compression model.

Source notebooks consolidated:
- z02_get_batch_aud_token.ipynb
- z02_get_batch_aud_token_single.ipynb
- z02_get_batch_aud_token_single_cut_token.ipynb

----------------------------------------
Usage:
    python get_tokens.py --input data/raw/ --output data/processed/tokens --model medium

Arguments:
    --input   Path to directory containing audio files (.wav, .mp3).
    --output  Path to directory where token files (*.npy) will be saved.
    --model   MusicGen model size: [small | medium | large] (default: medium).

Output:
    For each audio file, a numpy file will be created in the output directory:
    - <basename>_tokens.npy   (EnCodec tokens)

Example:
    python get_tokens.py --input ./data/raw/ --output ./data/processed/tokens --model small
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
    (based on z02_get_batch_aud_token*.ipynb)
    """
    wav, sr = torchaudio.load(file_path)
    if wav.shape[0] > 1:  # convert to mono
        wav = wav.mean(dim=0, keepdim=True)
    wav = convert_audio(wav, sr, sample_rate, mono=True)
    return wav, sample_rate


def extract_tokens(model, file_path: str, output_dir: str):
    """
    Extract EnCodec tokens for a single audio file and save to .npy.
    """
    os.makedirs(output_dir, exist_ok=True)

    # === Preprocess ===
    wav, sr = preprocess_audio(file_path)
    wav = wav.to(model.device)

    # === Encode to tokens ===
    with torch.no_grad():
        tokens = model.compression_model.encode(wav)

    tokens_np = tokens[0].cpu().numpy()

    # === Save ===
    base = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(output_dir, f"{base}_tokens.npy")
    np.save(out_path, tokens_np)
    print(f"[INFO] Saved tokens for {file_path} → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory containing audio files (wav, mp3)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory to save tokens")
    parser.add_argument("--model", type=str, default="medium",
                        choices=["small", "medium", "large"],
                        help="MusicGen model size (default=medium)")
    args = parser.parse_args()

    # === Load MusicGen compression model ===
    print(f"[INFO] Loading MusicGen model ({args.model}) ...")
    model = MusicGen.get_pretrained(args.model)
    model.eval()
    model.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(model.device)

    # === Process all files ===
    for fname in os.listdir(args.input):
        if fname.lower().endswith((".wav", ".mp3")):
            extract_tokens(model,
                           os.path.join(args.input, fname),
                           args.output)


if __name__ == "__main__":
    main()
