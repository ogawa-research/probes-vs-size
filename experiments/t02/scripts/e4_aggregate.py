"""T02 P0-c completion E4: aggregate per-model outputs into band_rule_outputs.json
(layer count L, >=0.9 longest run, adopted band, whether the threshold was
relaxed below 0.9, and capacity status) after e4_driver.sh has run.
"""
import json

MODELS = [
    "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b",
    "gemma-3-270m", "gemma-3-1b", "gemma-3-4b", "gemma-3-12b", "gemma-3-27b",
]

out = {}
for np_id in MODELS:
    entry = {"np_model_id": np_id}
    cka_path = f"/workspace/t02/cka_band_{np_id}.json"
    band_path = f"/workspace/t02/band_capacity_{np_id}.json"

    try:
        cka = json.load(open(cka_path))
        entry["cka_status"] = "ok"
        entry["n_layers"] = cka["n_layers"]
        entry["d_model"] = cka["d_model"]
        entry["cka_elapsed_s"] = cka.get("elapsed_s")
    except FileNotFoundError:
        entry["cka_status"] = "missing"
    except Exception as exc:
        entry["cka_status"] = f"error: {exc}"

    try:
        band = json.load(open(band_path))
        br = band.get("band_rule_result", {})
        entry["threshold_used"] = br.get("threshold_used")
        entry["threshold_relaxed_below_0.9"] = (br.get("threshold_used") is not None and br["threshold_used"] < 0.9)
        entry["longest_run"] = br.get("longest_run")
        entry["longest_run_length"] = br.get("longest_run_length")
        entry["floor_L_3"] = br.get("floor_L_3")
        entry["adopted_band"] = br.get("band")
        entry["capacity_status"] = band.get("capacity_status")
        entry["capacity_error"] = band.get("capacity_error")
    except FileNotFoundError:
        entry["band_status"] = "missing (cka likely failed -- see cka_status)"
    except Exception as exc:
        entry["band_status"] = f"error: {exc}"

    out[np_id] = entry

with open("/workspace/t02/band_rule_outputs.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
