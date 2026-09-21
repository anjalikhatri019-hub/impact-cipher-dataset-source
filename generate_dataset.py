"""Generate the original Impact Cipher synthetic elastic-collision corpus.

No external data, media, pretrained model, or downloaded asset is used.
The release seed is organizer-held; publishing this source does not publish it.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from itertools import islice, repeat
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VERSION = "1.0"
N = 10
PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
CAL_TRIALS = 4
OBS_STEPS = 160
QUERY_STEPS = 192
DT = 0.025
SUBSTEPS = 2
FAMILIES = 30
SYSTEMS_PER_FAMILY = 80
TEST_FAMILIES = frozenset(range(25, 30))
VALIDATION_FAMILIES = frozenset(range(20, 25))


def _family_masses(family: int, rng: np.random.Generator) -> np.ndarray:
    x = np.arange(N, dtype=np.float64)
    a = 0.30 + 0.04 * (family % 5)
    b = 0.16 + 0.025 * ((family // 5) % 4)
    phase = (family * 0.61803398875) % 1.0 * 2 * np.pi
    base = np.exp(a * np.sin(2 * np.pi * x / N + phase) + b * np.cos(4 * np.pi * x / N - phase))
    base *= np.exp(rng.normal(0.0, 0.045, N))
    base *= 1.5 / np.exp(np.mean(np.log(base)))
    # Quantize before simulation so the hidden audit mass vector exactly defines
    # the benchmark dynamics; there is no unrecoverable float64 residue.
    return base[rng.permutation(N)].astype(np.float32).astype(np.float64)


def _initial_state(rng: np.random.Generator, radii: np.ndarray, masses: np.ndarray, speed: float) -> np.ndarray:
    positions: list[np.ndarray] = []
    for i in range(N):
        for _ in range(10_000):
            candidate = rng.uniform(radii[i] + 0.015, 1.0 - radii[i] - 0.015, 2)
            if all(np.linalg.norm(candidate - positions[j]) > radii[i] + radii[j] + 0.018 for j in range(i)):
                positions.append(candidate)
                break
        else:
            raise RuntimeError("could not place non-overlapping discs")
    velocity = rng.normal(0.0, speed, (N, 2))
    velocity -= np.sum(masses[:, None] * velocity, axis=0) / np.sum(masses)
    norm = np.linalg.norm(velocity, axis=1)
    velocity *= np.minimum(1.0, 0.72 / np.maximum(norm, 1e-9))[:, None]
    # Initial states are quantized before simulation and publication.
    return np.concatenate([np.asarray(positions), velocity], axis=1).astype(np.float32)


def simulate(initial: np.ndarray, radii: np.ndarray, masses: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    state = initial.astype(np.float64, copy=True)
    positions, velocity = state[:, :2], state[:, 2:]
    trajectory = np.empty((steps, N, 4), dtype=np.float32)
    impulses = np.zeros(len(PAIRS), dtype=np.float64)
    h = DT / SUBSTEPS
    for step in range(steps):
        for _ in range(SUBSTEPS):
            positions += h * velocity
            for axis in range(2):
                low = positions[:, axis] < radii
                positions[low, axis] = 2 * radii[low] - positions[low, axis]
                velocity[low, axis] = np.abs(velocity[low, axis])
                high = positions[:, axis] > 1.0 - radii
                positions[high, axis] = 2 * (1.0 - radii[high]) - positions[high, axis]
                velocity[high, axis] = -np.abs(velocity[high, axis])
            for pair_index, (i, j) in enumerate(PAIRS):
                delta = positions[i] - positions[j]
                distance = float(np.linalg.norm(delta))
                contact = radii[i] + radii[j]
                if distance >= contact:
                    continue
                if distance < 1e-12:
                    direction = np.array([1.0, 0.0])
                else:
                    direction = delta / distance
                overlap = contact - distance
                positions[i] += direction * overlap * (masses[j] / (masses[i] + masses[j]))
                positions[j] -= direction * overlap * (masses[i] / (masses[i] + masses[j]))
                relative_normal = float(np.dot(velocity[i] - velocity[j], direction))
                if relative_normal >= 0:
                    continue
                impulse = -2.0 * relative_normal / (1.0 / masses[i] + 1.0 / masses[j])
                velocity[i] += (impulse / masses[i]) * direction
                velocity[j] -= (impulse / masses[j]) * direction
                impulses[pair_index] += impulse
        trajectory[step, :, :2] = positions
        trajectory[step, :, 2:] = velocity
    return trajectory, impulses.astype(np.float32)


def _observed_calibration(rng: np.random.Generator, masses: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initials = np.empty((CAL_TRIALS, N, 4), dtype=np.float32)
    observed = np.empty((CAL_TRIALS, OBS_STEPS, N, 4), dtype=np.float32)
    valid = np.ones((CAL_TRIALS, OBS_STEPS, N), dtype=np.uint8)
    for trial in range(CAL_TRIALS):
        initial = _initial_state(rng, radii, masses, rng.uniform(0.30, 0.46))
        clean, _ = simulate(initial, radii, masses, OBS_STEPS)
        noisy = clean.copy()
        noisy[:, :, :2] += rng.normal(0.0, rng.uniform(0.0012, 0.0032), noisy[:, :, :2].shape)
        noisy[:, :, 2:] += rng.normal(0.0, rng.uniform(0.009, 0.024), noisy[:, :, 2:].shape)
        miss_rate = rng.uniform(0.10, 0.26)
        trial_valid = rng.random((OBS_STEPS, N)) >= miss_rate
        for body in range(N):
            for _ in range(2):
                start = int(rng.integers(8, OBS_STEPS - 16))
                trial_valid[start:start + int(rng.integers(3, 12)), body] = False
        noisy[~trial_valid] = np.nan
        initials[trial] = initial
        observed[trial] = noisy
        valid[trial] = trial_valid
    return initials, observed, valid


def generate_one(index: int, master_seed: int) -> dict[str, object]:
    family = index // SYSTEMS_PER_FAMILY
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, index, 0xE1A57C]))
    masses = _family_masses(family, rng)
    radii = rng.uniform(0.032, 0.052, N).astype(np.float32).astype(np.float64)
    cal_initial, cal_state, valid = _observed_calibration(rng, masses, radii)
    for _ in range(128):
        query_initial = _initial_state(rng, radii, masses, rng.uniform(0.34, 0.52))
        query_trajectory, impulses = simulate(query_initial, radii, masses, QUERY_STEPS)
        if np.count_nonzero(impulses > 1e-5) >= 6:
            break
    else:
        raise RuntimeError("could not generate sufficiently interactive query")
    target = np.concatenate([impulses, query_trajectory[-1, :, 2:].reshape(-1)]).astype(np.float32)
    opaque_id = "ic_" + hashlib.sha256(f"impact-cipher:{master_seed}:{index}".encode()).hexdigest()[:24]
    split = "test" if family in TEST_FAMILIES else "train"
    return {
        "sample_id": opaque_id,
        "split": split,
        "family_id": family,
        "validation_fold": int(family in VALIDATION_FAMILIES),
        "mass_bytes": masses.astype("<f4").tobytes(),
        "radii_bytes": radii.astype("<f4").tobytes(),
        "calibration_initial_bytes": cal_initial.astype("<f4").tobytes(),
        "calibration_state_bytes": cal_state.astype("<f4").tobytes(),
        "calibration_valid_bytes": valid.tobytes(),
        "query_initial_bytes": query_initial.astype("<f4").tobytes(),
        "target_values": " ".join(f"{float(value):.8g}" for value in target),
    }


def generate(out: Path, seed: int, count: int = FAMILIES * SYSTEMS_PER_FAMILY, workers: int = 1) -> None:
    if count != FAMILIES * SYSTEMS_PER_FAMILY:
        raise ValueError(f"release count must be {FAMILIES * SYSTEMS_PER_FAMILY}")
    out.mkdir(parents=True, exist_ok=True)
    target = out / "impact_cipher_raw.parquet"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    writer = None
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        records = (executor.map(generate_one, range(count), repeat(seed), chunksize=1)
                   if executor is not None else (generate_one(i, seed) for i in range(count)))
        for start in range(0, count, 24):
            table = pa.Table.from_pylist(list(islice(records, 24)))
            if writer is None:
                writer = pq.ParquetWriter(target, table.schema, compression=None, use_dictionary=False)
            writer.write_table(table)
            if start % 240 == 0:
                print(f"generated {min(start + 24, count)}/{count}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    metadata = {
        "version": VERSION,
        "rows": count,
        "train_rows": 2000,
        "test_rows": 400,
        "seed_sha256": hashlib.sha256(str(seed).encode()).hexdigest(),
        "raw_sha256": hashlib.file_digest(target.open("rb"), "sha256").hexdigest(),
        "calibration_shape": [CAL_TRIALS, OBS_STEPS, N, 4],
        "target_values": len(PAIRS) + 2 * N,
    }
    (out / "generation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=FAMILIES * SYSTEMS_PER_FAMILY)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    generate(args.out, args.seed, args.count, args.workers)
