"""T02 P0-c completion E5: gemma-3-270m lens quality check.

Fetches convergence.csv for the gemma-3-270m lens from HF Hub and records
the identity_distance trajectory (converged vs still-declining vs diverging),
then cross-references E4's gemma-3-270m CKA/capacity numbers against the
other E4 models for an order-of-magnitude comparison. Records facts only --
no pass/fail judgment (researcher's call per the instructions).
"""
import json

from huggingface_hub import hf_hub_download

REPO = "neuronpedia/jacobian-lens"
CONVERGENCE_PATH = "gemma-3-270m/jlens/Salesforce-wikitext/gemma-3-270m_convergence.csv"
OUT_PATH = "/workspace/t02/gemma270m_quality_check.json"

OTHER_MODELS = ["qwen3-4b", "qwen3-8b", "qwen3-14b", "gemma-3-1b", "gemma-3-4b", "gemma-3-12b"]


def main():
    result = {}

    try:
        local = hf_hub_download(REPO, CONVERGENCE_PATH)
        text = open(local).read()
        lines = [l for l in text.splitlines() if l.strip()]
        result["convergence_csv_found"] = True
        result["convergence_csv_header"] = lines[0] if lines else None
        result["convergence_csv_n_rows"] = max(0, len(lines) - 1)
        result["convergence_csv_first_5_rows"] = lines[1:6]
        result["convergence_csv_last_5_rows"] = lines[-5:] if len(lines) > 5 else lines[1:]
        result["convergence_csv_raw_path"] = local
    except Exception as exc:
        result["convergence_csv_found"] = False
        result["convergence_csv_error"] = f"{type(exc).__name__}: {exc}"

    # cross-reference with E4 outputs, if present
    try:
        band_rule_outputs = json.load(open("/workspace/t02/band_rule_outputs.json"))
    except FileNotFoundError:
        band_rule_outputs = None
        result["e4_cross_reference"] = "band_rule_outputs.json not found -- run E4 first"

    if band_rule_outputs is not None:
        gemma_270m = band_rule_outputs.get("gemma-3-270m", {})
        others = {m: band_rule_outputs.get(m, {}) for m in OTHER_MODELS}
        result["e4_cross_reference"] = {
            "gemma-3-270m": {
                "cka_status": gemma_270m.get("cka_status"),
                "capacity_status": gemma_270m.get("capacity_status"),
                "adopted_band": gemma_270m.get("adopted_band"),
                "threshold_used": gemma_270m.get("threshold_used"),
                "longest_run_length": gemma_270m.get("longest_run_length"),
                "n_layers": gemma_270m.get("n_layers"),
            },
            "other_models_summary": {
                m: {
                    "cka_status": v.get("cka_status"),
                    "capacity_status": v.get("capacity_status"),
                    "threshold_used": v.get("threshold_used"),
                    "longest_run_length": v.get("longest_run_length"),
                    "n_layers": v.get("n_layers"),
                }
                for m, v in others.items()
            },
        }

    # known fact from P0-e config.yaml (already read 2026-07-10): recorded here for
    # convenience, not re-fetched (config.yaml result field is static metadata).
    result["known_final_identity_distance"] = {
        "gemma-3-270m": 3.509132,
        "gemma-3-4b (reference, other model)": 0.684409,
        "note": "from config.yaml results.final_identity_distance, read directly during E4 prep; "
                "gemma-3-270m's value is ~5-9x larger than the other Gemma-3 sizes' fitted lenses",
    }

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
