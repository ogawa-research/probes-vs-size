"""Shared band-detection machine rule for T02 P0-c completion (design v1.4 S2 / S11'''-2).

band = front floor(L/3) layers of the longest contiguous run of adjacent-layer
CKA >= threshold (default 0.9; lowered by 0.05 steps if the longest run at 0.9
has fewer than 3 layers).
"""
import numpy as np


def longest_run_at_threshold(cka, threshold):
    n = cka.shape[0]
    best = [0]
    cur = [0]
    for i in range(1, n):
        if cka[i - 1, i] >= threshold:
            cur.append(i)
        else:
            if len(cur) > len(best):
                best = cur
            cur = [i]
    if len(cur) > len(best):
        best = cur
    return best


def compute_band(cka_matrix, n_layers_for_floor, start_threshold=0.9, threshold_step=0.05, min_run_length=3):
    """cka_matrix: NxN nested list/array of pairwise linear CKA.
    n_layers_for_floor: L used for floor(L/3) (lens source-layer count).
    Returns dict with threshold_used, longest_run, band, and the search trail.
    """
    cka = np.array(cka_matrix, dtype=np.float64)
    threshold = start_threshold
    run = longest_run_at_threshold(cka, threshold)
    thresholds_tried = [threshold]
    while len(run) < min_run_length and threshold - threshold_step > 0:
        threshold = round(threshold - threshold_step, 2)
        run = longest_run_at_threshold(cka, threshold)
        thresholds_tried.append(threshold)

    k = max(1, n_layers_for_floor // 3)
    band = run if len(run) <= k else run[:k]

    return {
        "threshold_used": threshold,
        "thresholds_tried": thresholds_tried,
        "longest_run": run,
        "longest_run_length": len(run),
        "n_layers_for_floor": n_layers_for_floor,
        "floor_L_3": k,
        "band": band,
    }


if __name__ == "__main__":
    # self-check against the already-established Qwen3-1.7B result (band 14-22).
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/t02/cka_band_qwen17b.json"
    data = json.load(open(path))
    cka = data["cka_matrix"]
    n_layers = data["n_layers"]
    result = compute_band(cka, n_layers_for_floor=n_layers - 1)  # lens source layers = n_layers-1 (excludes final)
    print(json.dumps(result, indent=2))
