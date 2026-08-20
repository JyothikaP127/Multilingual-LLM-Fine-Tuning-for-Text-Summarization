"""Controlled max_source_length analysis: 512 vs 768 vs 1024.

Reuses the exact same 2,800-example-per-language subsample (train+val+test,
same seed/revision) that scripts/inspect_dataset.py already pulled and
cached locally -- no additional rows are downloaded. Does not train
anything and does not modify configs/data.yaml (the active
max_source_length stays at 512 until an explicit decision is made).

Writes:
  results/qa/truncation_comparison.csv
  results/qa/truncation_comparison.json
  results/qa/truncation_comparison.png  (truncation % vs. max_source_length)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.token_stats import compute_stats_at_thresholds, tokenize_lengths  # noqa: E402
from src.data.xlsum_loader import load_balanced_subsample  # noqa: E402
from src.utils.config import load_yaml, repo_path  # noqa: E402
from src.utils.reporting import save_json_report  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

THRESHOLDS = [512, 768, 1024]


def main() -> None:
    base_cfg = load_yaml("base.yaml")
    data_cfg = load_yaml("data.yaml")
    set_seed(base_cfg["seed"])

    revision = data_cfg["dataset"]["revision"]
    seed = data_cfg["dataset"]["sampling_seed"]
    split_sizes = data_cfg["dataset"]["split_sizes"]
    languages = [lang["config_name"] for lang in data_cfg["dataset"]["languages"]]

    print(f"=== max_source_length analysis: {THRESHOLDS} ===")
    print("Reusing cached subsample (no new downloads); active config max_source_length "
          f"remains {data_cfg['preprocessing']['max_source_length']} (unchanged).\n")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    csv_rows: list[dict] = []
    json_report: dict = {"thresholds": THRESHOLDS, "languages": {}}
    plot_data: dict[str, dict[int, float]] = {}

    for lang in languages:
        splits = {
            split: load_balanced_subsample(lang, split, n, seed, revision)
            for split, n in split_sizes.items()
        }
        all_rows = [row for rows in splits.values() for row in rows]
        source_lengths = tokenize_lengths(tokenizer, [row["text"] for row in all_rows])

        stats_by_threshold = compute_stats_at_thresholds(source_lengths, "source_text", THRESHOLDS)

        print(f"--- {lang} (n={len(all_rows)}) ---")
        json_report["languages"][lang] = {}
        plot_data[lang] = {}

        for threshold in THRESHOLDS:
            stats = stats_by_threshold[threshold]
            fits_pct = 100.0 - stats.truncated_pct
            print(
                f"  max_source_length={threshold:4d}: truncated={stats.truncated_pct:5.1f}% "
                f"fits={fits_pct:5.1f}% | mean={stats.mean:.1f} median={stats.median:.0f} "
                f"p90={stats.p90:.0f} p95={stats.p95:.0f} max={stats.max}"
            )
            csv_rows.append(
                {
                    "language": lang,
                    "max_source_length": threshold,
                    "n_examples": stats.n,
                    "truncated_count": stats.truncated_count,
                    "truncated_pct": round(stats.truncated_pct, 2),
                    "fits_completely_pct": round(fits_pct, 2),
                    "mean_tokens": round(stats.mean, 2),
                    "median_tokens": stats.median,
                    "p90_tokens": stats.p90,
                    "p95_tokens": stats.p95,
                    "max_tokens": stats.max,
                }
            )
            json_report["languages"][lang][str(threshold)] = {
                **stats.to_dict(),
                "fits_completely_pct": fits_pct,
            }
            plot_data[lang][threshold] = stats.truncated_pct
        print()

    # --- CSV ---
    csv_path = repo_path("results/qa/truncation_comparison.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"CSV written to: {csv_path}")

    # --- JSON ---
    json_path = save_json_report(json_report, repo_path("results/qa/truncation_comparison.json"))
    print(f"JSON written to: {json_path}")

    # --- Plot ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"english": "o", "hindi": "s", "telugu": "^"}
    for lang in languages:
        ys = [plot_data[lang][t] for t in THRESHOLDS]
        ax.plot(THRESHOLDS, ys, marker=markers.get(lang, "o"), label=lang.capitalize())

    ax.set_xlabel("max_source_length (tokens)")
    ax.set_ylabel("Examples truncated (%)")
    ax.set_title("Source truncation rate vs. max_source_length, by language")
    ax.set_xticks(THRESHOLDS)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = repo_path("results/qa/truncation_comparison.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Plot written to: {plot_path}")


if __name__ == "__main__":
    main()
