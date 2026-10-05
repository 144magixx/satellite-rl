"""S1-03-R1 read-only non-STOP action acceptance curve.

This companion reads the completed training actions once. It does not reload
episodes, a policy, or the experiment environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED_RUN_ID = "train_ppo_20261005_093803_4570222a"
EXPECTED_BUDGET = 262144
EXPECTED_NON_STOP = 261602
EXPECTED_ACCEPTED = 149062
EXPECTED_STOP = 542
DEFAULT_BIN_SIZE = 16384


def require(condition, message):
    if not condition:
        raise ValueError(message)


def accumulate(actions, budget=EXPECTED_BUDGET, bin_size=DEFAULT_BIN_SIZE):
    require(budget > 0 and bin_size > 0 and budget % bin_size == 0,
            "budget must be a positive multiple of bin size")
    bins = [{"bin_index": i + 1, "env_step_start": i * bin_size + 1,
             "env_step_end": (i + 1) * bin_size, "all_actions": 0,
             "non_stop": 0, "accepted_non_stop": 0, "stop": 0}
            for i in range(budget // bin_size)]
    count = 0
    for action in actions:
        step = action.get("env_step")
        require(type(step) is int and step == count + 1 and step <= budget,
                f"action env_step is missing, duplicated, or out of order at {count + 1}")
        fields = action.get("requested_action_fields")
        require(isinstance(fields, dict) and isinstance(fields.get("operation"), str),
                f"missing requested operation at {step}")
        accepted = action.get("accepted")
        require(type(accepted) is bool, f"missing accepted flag at {step}")
        bucket = bins[(step - 1) // bin_size]
        bucket["all_actions"] += 1
        if fields["operation"] == "stop":
            require(not accepted, f"STOP marked accepted at {step}")
            bucket["stop"] += 1
        else:
            bucket["non_stop"] += 1
            bucket["accepted_non_stop"] += int(accepted)
        count += 1
    require(count == budget, f"action log has {count} rows, expected {budget}")
    for bucket in bins:
        require(bucket["all_actions"] == bin_size and
                bucket["all_actions"] == bucket["non_stop"] + bucket["stop"],
                f"incomplete or inconsistent bin {bucket['bin_index']}")
        bucket["acceptance_rate_non_stop"] = (bucket["accepted_non_stop"] / bucket["non_stop"]
                                               if bucket["non_stop"] else None)
    return bins


def read_hashed_actions(path, digest):
    with path.open("rb") as source:
        for line in source:
            digest.update(line)
            if line.strip():
                yield json.loads(line)


def plot(path, bins, font_path):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    require(font_path is not None and font_path.is_file(), "pass an existing --font with Chinese glyphs")
    font_manager.fontManager.addfont(str(font_path))
    glyphs = font_manager.ft2font.FT2Font(str(font_path)).get_charmap()
    require(all(ord(c) in glyphs for c in "训练实际交互步非停止动作接受率原始区间数量"),
            "font lacks required Chinese glyphs")
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["font.sans-serif"] = [family]
    plt.rcParams["axes.unicode_minus"] = False
    steps = [row["env_step_end"] for row in bins]
    rates = [100 * row["acceptance_rate_non_stop"] for row in bins]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    top.plot(steps, rates, marker="o", label="P1 PPO：非STOP动作接受率（每16384交互）")
    top.set_ylabel("接受率（%）")
    top.set_title("S1-03-R1 训练动作接受率，seed 51001，原始动作账分箱")
    top.set_ylim(0, 100)
    top.legend()
    bottom.plot(steps, [row["non_stop"] for row in bins], marker="s", label="非STOP请求数")
    bottom.plot(steps, [row["accepted_non_stop"] for row in bins], marker="^", label="接受数")
    bottom.plot(steps, [row["stop"] for row in bins], marker=".", label="STOP数")
    bottom.set_xlabel("实际环境交互步（区间终点）")
    bottom.set_ylabel("动作数/区间")
    bottom.legend()
    for ax in (top, bottom):
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build(job_dir: Path, output: Path, font_path: Path, bin_size=DEFAULT_BIN_SIZE):
    require(not output.exists(), "output exists; choose a new derived directory")
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    require(job.get("run_id") == EXPECTED_RUN_ID and status.get("run_id") == EXPECTED_RUN_ID,
            "wrong training run")
    require(status.get("stage") == "completed" and status.get("env_step") == EXPECTED_BUDGET and
            status.get("budget_env_steps") == EXPECTED_BUDGET,
            "training did not complete the fixed budget")
    digest = hashlib.sha256()
    bins = accumulate(read_hashed_actions(job_dir / "actions.jsonl", digest),
                      EXPECTED_BUDGET, bin_size)
    totals = {field: sum(row[field] for row in bins)
              for field in ("all_actions", "non_stop", "accepted_non_stop", "stop")}
    require(totals == {"all_actions": EXPECTED_BUDGET, "non_stop": EXPECTED_NON_STOP,
                       "accepted_non_stop": EXPECTED_ACCEPTED, "stop": EXPECTED_STOP},
            f"training action account mismatch: {totals}")
    output.mkdir(parents=True)
    with (output / "action_acceptance_bins.csv").open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(bins[0]))
        writer.writeheader()
        writer.writerows(bins)
    summary = {"run_id": EXPECTED_RUN_ID, "science_commit": job["git_commit"],
               "actions_sha256": digest.hexdigest(), "budget_env_steps": EXPECTED_BUDGET,
               "bin_size": bin_size, "bin_count": len(bins), "totals": totals,
               "overall_non_stop_acceptance_rate": totals["accepted_non_stop"] / totals["non_stop"],
               "denominator": "non_STOP_requested_actions; STOP excluded",
               "source": str(job_dir / "actions.jsonl"), "bins": bins}
    (output / "action_acceptance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot(output / "中文训练非STOP动作接受率.png", bins, font_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True, help="completed R1 train job directory")
    parser.add_argument("--output", type=Path, required=True, help="new derived output directory")
    parser.add_argument("--font", type=Path, required=True, help="Chinese TTF/TTC font")
    parser.add_argument("--bin-size", type=int, default=DEFAULT_BIN_SIZE)
    args = parser.parse_args()
    result = build(args.job_dir, args.output, args.font, args.bin_size)
    print(json.dumps({"output": str(args.output), "actions_sha256": result["actions_sha256"],
                      "totals": result["totals"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
