#!/usr/bin/env python3
"""
process_results.py -- SDM-focused feature extraction

This script focuses exclusively on computing the Sample Distance Matrix (SDM)
as described in the ICASSP 2026 paper "Detecting Training Data in Music
Generation Models with Sample Distance Matrix". It reads per-sample logits
saved as NumPy arrays (produced by run_musicgen.py), optionally shuffles
their time indices, chunks each sample into non-overlapping blocks, computes
chunk-level representations, and then computes pairwise sample distances by
averaging chunk-to-chunk cosine distances between samples.

Outputs:
- distances_sdm.npy : NxN numpy array of pairwise cosine distances
- sample_vectors.npy : NxD numpy array of per-sample aggregated vectors (optional)
- names.txt          : text file listing sample basenames in the same order

Source of implementation ideas (extracted from user notebooks):
- MIA_on_musicgen/z03_get_batch_logit.ipynb
- MIA_on_musicgen/z04_result_data_processing.ipynb
- MIA_on_musicgen/z04_result_data_processing_single.ipynb
- MIA_on_musicgen/z05_show_result_frame_chunk2_auc.ipynb
- MIA_on_musicgen/z05_show_result_frame_chunk2_tsne.ipynb

Usage example:
    python process_results.py --input data/processed/logits --output data/processed/sdm --chunk_size 100 --shuffle --seed 42

Notes:
- Input files expected: <basename>_logits.npy (per-timestep logits; shape: [T, C] or similar)
- The script is intentionally focused and does NOT implement other MIA features.
"""

import os
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

def load_logits_files(input_dir):
    """
    Load *_logits.npy files from input_dir, return list of (basename, logits_array).
    """
    files = sorted([f for f in os.listdir(input_dir) if f.endswith("_logits.npy")])
    names = [f[:-len("_logits.npy")] for f in files]
    arrays = []
    for f in files:
        path = os.path.join(input_dir, f)
        arr = np.load(path, allow_pickle=False)
        # Ensure shape is (T, C)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        arrays.append(arr)
    return names, arrays

def shuffle_time_indices(logits, seed=None):
    """
    Shuffle time indices of a logits array (in-place copy returned).
    logits: np.ndarray of shape (T, C)
    """
    rng = np.random.default_rng(seed)
    T = logits.shape[0]
    perm = rng.permutation(T)
    return logits[perm]

def chunk_logits(logits, chunk_size):
    """
    Split logits sequence into non-overlapping chunks of size chunk_size.
    Discard remainder shorter than chunk_size.
    Return array of shape (num_chunks, chunk_size, C).
    """
    T = logits.shape[0]
    num_chunks = T // chunk_size
    if num_chunks == 0:
        return np.empty((0, chunk_size, logits.shape[1]), dtype=logits.dtype)
    trimmed = logits[:num_chunks * chunk_size]
    return trimmed.reshape(num_chunks, chunk_size, logits.shape[1])

def chunk_level_vectors(chunks, pooling='mean'):
    """
    Convert chunk array (num_chunks, chunk_size, C) into chunk-level vectors
    shape (num_chunks, C) by pooling across time within each chunk.
    pooling: 'mean' or 'max' (mean recommended).
    """
    if chunks.size == 0:
        return np.empty((0, chunks.shape[-1]), dtype=chunks.dtype)
    if pooling == 'mean':
        return chunks.mean(axis=1)
    elif pooling == 'max':
        return chunks.max(axis=1)
    else:
        raise ValueError("Unsupported pooling: %s" % pooling)

def sample_pair_distance(chunks_a, chunks_b):
    """
    Given chunk-level vectors for sample a (Na x D) and sample b (Nb x D),
    compute the average cosine distance between all chunk pairs.
    If one of the samples has zero chunks, returns np.nan.
    """
    if chunks_a.shape[0] == 0 or chunks_b.shape[0] == 0:
        return np.nan
    # compute pairwise cosine distances between all chunks (Na*Nb distances)
    d = cosine_distances(chunks_a, chunks_b)
    return float(d.mean())

def compute_sdm(names, logits_list, chunk_size=100, shuffle=False, seed=None, pooling='mean', normalize_vectors=False, save_vectors=False, verbose=True):
    """
    Compute SDM distance matrix.
    Steps per sample:
      1) optionally shuffle time indices
      2) split into non-overlapping chunks of chunk_size
      3) pool each chunk into a vector (mean)
    Then for every pair (i, j) compute mean cosine distance between chunks_i and chunks_j.
    Returns:
      - distances: NxN numpy array
      - sample_vectors: list of arrays with shape (num_chunks_i, D)
    """
    N = len(names)
    sample_chunks = []  # list of (num_chunks_i, D) arrays
    rng = np.random.default_rng(seed)

    # 1) build chunk-level vectors for each sample
    for idx, logits in enumerate(tqdm(logits_list, desc="preprocessing samples")):
        arr = np.array(logits, dtype=float)
        if shuffle:
            # use sample-specific seed for reproducibility
            arr = shuffle_time_indices(arr, seed=(None if seed is None else int(seed + idx)))
        chunks = chunk_logits(arr, chunk_size)
        chunk_vecs = chunk_level_vectors(chunks, pooling=pooling)  # (num_chunks, D)
        sample_chunks.append(chunk_vecs)

    # Optionally normalize chunk vectors per vector (L2)
    if normalize_vectors:
        for i in range(N):
            if sample_chunks[i].size == 0:
                continue
            norms = np.linalg.norm(sample_chunks[i], axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            sample_chunks[i] = sample_chunks[i] / norms

    # Prepare distances matrix
    distances = np.zeros((N, N), dtype=float)
    # compute upper triangle and mirror (distance matrix is symmetric)
    for i in tqdm(range(N), desc="computing pairwise distances"):
        for j in range(i, N):
            d = sample_pair_distance(sample_chunks[i], sample_chunks[j])
            distances[i, j] = d
            distances[j, i] = d

    # Optionally compute a per-sample aggregated vector (mean of chunk vectors)
    sample_vectors = None
    if save_vectors:
        vecs = []
        for chunk_vecs in sample_chunks:
            if chunk_vecs.shape[0] == 0:
                vecs.append(np.zeros((1, logits_list[0].shape[1]), dtype=float)[0])
            else:
                vecs.append(chunk_vecs.mean(axis=0))
        sample_vectors = np.stack(vecs, axis=0)

    return distances, sample_vectors

def save_outputs(output_dir, names, distances, sample_vectors=None, save_csv=True):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "distances_sdm.npy"), distances)
    if save_csv:
        # Save CSV with header of names for readability (floating distances)
        import csv
        csv_path = os.path.join(output_dir, "distances_sdm.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([""] + names)
            for name, row in zip(names, distances):
                writer.writerow([name] + list(row))
    if sample_vectors is not None:
        np.save(os.path.join(output_dir, "sample_vectors.npy"), sample_vectors)
    # save names order
    with open(os.path.join(output_dir, "names.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(names))

def main():
    parser = argparse.ArgumentParser(description="Compute Sample Distance Matrix (SDM) from logits files.")
    parser.add_argument("--input", required=True, help="Directory containing *_logits.npy files (produced by run_musicgen.py)")
    parser.add_argument("--output", required=True, help="Directory to save SDM outputs")
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of timesteps per chunk (default:100)")
    parser.add_argument("--shuffle", action="store_true", help="Randomly shuffle time indices before chunking (recommended in paper)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for shuffling (default:1234)")
    parser.add_argument("--pooling", choices=["mean", "max"], default="mean", help="Pooling within chunk to produce chunk vector (default: mean)")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize chunk vectors before computing distances")
    parser.add_argument("--save_vectors", action="store_true", help="Also save per-sample aggregated vectors (sample_vectors.npy)")
    parser.add_argument("--no_csv", action="store_true", help=\"\"\"Do not save CSV (only .npy). CSV can be large for many samples.\"\"\")
    args = parser.parse_args()

    names, logits_list = load_logits_files(args.input)
    if len(names) == 0:
        raise SystemExit(f"No *_logits.npy files found in {args.input}")

    distances, sample_vectors = compute_sdm(names, logits_list,
                                            chunk_size=args.chunk_size,
                                            shuffle=args.shuffle,
                                            seed=args.seed,
                                            pooling=args.pooling,
                                            normalize_vectors=args.normalize,
                                            save_vectors=args.save_vectors,
                                            verbose=True)

    save_outputs(args.output, names, distances, sample_vectors, save_csv=not args.no_csv)
    print(f"[INFO] SDM computation finished. Saved to {args.output}")

if __name__ == "__main__":
    main()
