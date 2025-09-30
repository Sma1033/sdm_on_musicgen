#!/usr/bin/env python3
"""
get_logits.py

Extract logits from MusicGen language model (LM) given audio files or tokens.

Source notebooks consolidated:
- z03_get_batch_logit.ipynb
- z03_get_batch_logit_single.ipynb
- z03_get_batch_logit_single_disrupted.ipynb

----------------------------------------
Usage:
    # Directly from audio
    python get_logits.py --input data/raw/ --output data/processed/logits --model medium --from_audio

    # From pre-extracted tokens
    python get_logits.py --input data/processed/tokens --output data/processed/logits --model medium

Arguments:
    --input      Directory containing audio files (.wav, .mp3) OR token files (*.npy).
    --output     Directory to save logits (*.npy).
    --model      MusicGen model size: [small | medium | large] (default: medium).
    --from_audio If set, will read audio and extract tokens automatically.
                 Otherwise expects *_tokens.npy files.

Output:
    For each input file, saves:
    - <basename>_logits.npy   (per-timestep logits from LM)

Example:
    python get_logits.py --input ./data/processed/tokens --output ./data/processed/logits --model medium
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
    (based on z03_get_batch_logit*.ipynb)
    """
    wav, sr = torchaudio.load(file_path)
    if wav.shape[0] > 1:  # convert to mono
        wav = wav.mean(dim=0, keepdim=True)
    wav = convert_audio(wav, sr, sample_rate, mono=True)
    return wav, sample_rate


def extract_logits_from_audio(model, file_path: str, output_dir: str):
    """
    Extract logits for a single audio file.
    """
    os.makedirs(output_dir, exist_ok=True)

    wav, sr = preprocess_audio(file_path)
    wav = wav.to(model.device)

    # Get tokens
    tokens = model.compression_model.encode(wav)

    # Get logits from LM
    with torch.no_grad():
        out = model.lm.forward(tokens)

    logits = out["logits"].cpu().numpy()

    base = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(output_dir, f"{base}_logits.npy")
    np.save(out_path, logits)
    print(f"[INFO] Saved logits for {file_path} → {out_path}")


def extract_logits_from_tokens(model, file_path: str, output_dir: str):
    """
    Extract logits for a single token file (*.npy).
    """
    os.makedirs(output_dir, exist_ok=True)

    tokens = np.load(file_path)
    tokens = torch.tensor(tokens, dtype=torch.long, device=model.device).unsqueeze(0)

    with torch.no_grad():
        out = model.lm.forward(tokens)

    logits = out["logits"].cpu().numpy()

    base = os.path.splitext(os.path.basename(file_path))[0].replace("_tokens", "")
    out_path = os.path.join(output_dir, f"{base}_logits.npy")
    np.save(out_path, logits)
    print(f"[INFO] Saved logits for {file_path} → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory containing audio or token files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory to save logits")
    parser.add_argument("--model", type=str, default="medium",
                        choices=["small", "medium", "large"],
                        help="MusicGen model size (default=medium)")
    parser.add_argument("--from_audio", action="store_true",
                        help="If set, will read audio and extract tokens automatically.")
    args = parser.parse_args()

    # === Load MusicGen ===
    print(f"[INFO] Loading MusicGen model ({args.model}) ...")
    model = MusicGen.get_pretrained(args.model)
    model.eval()
    model.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(model.device)

    # === Process files ===
    for fname in os.listdir(args.input):
        path = os.path.join(args.input, fname)
        if args.from_audio and fname.lower().endswith((".wav", ".mp3")):
            extract_logits_from_audio(model, path, args.output)
        elif (not args.from_audio) and fname.endswith("_tokens.npy"):
            extract_logits_from_tokens(model, path, args.output)


if __name__ == "__main__":
    main()
