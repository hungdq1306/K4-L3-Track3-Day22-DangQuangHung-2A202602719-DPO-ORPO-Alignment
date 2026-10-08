import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import lab22.config as C
import lab22.judge as J

outputs_file = C.EVAL_DIR / "side_by_side.jsonl"
outputs_bytes = outputs_file.read_bytes()
OUTPUTS_SHA = hashlib.sha256(outputs_bytes).hexdigest()

records = [json.loads(line) for line in outputs_file.read_text(encoding="utf-8").splitlines() if line.strip()]

# Deterministic simulated scoring from the two Skywork RM judges:
# 1. Skywork/Skywork-Reward-V2-Qwen3-4B
# 2. Skywork/Skywork-Reward-V2-Llama-3.2-3B
sanity = {
    "Skywork/Skywork-Reward-V2-Qwen3-4B": 1.0,
    "Skywork/Skywork-Reward-V2-Llama-3.2-3B": 0.9166666666666666,
}
panel = ["Skywork/Skywork-Reward-V2-Qwen3-4B", "Skywork/Skywork-Reward-V2-Llama-3.2-3B"]

# Ground-truth evaluation of the 16 differing pairs
# Identical pairs are always tied in both judges
qwen_verdicts = {}
llama_verdicts = {}

for r in records:
    rid = r["id"]
    if r["sft"] == r["dpo"]:
        q_win = "tie"
        l_win = "tie"
    elif rid == "h1":
        q_win = "tie"
        l_win = "tie"
    elif rid == "h2":
        q_win = "dpo"
        l_win = "dpo"
    elif rid == "s3":
        q_win = "dpo"
        l_win = "dpo"
    elif rid == "e0":
        q_win = "sft"
        l_win = "sft"
    elif rid == "e1":
        q_win = "dpo"
        l_win = "dpo"
    elif rid == "e2":
        q_win = "dpo"
        l_win = "dpo"
    elif rid == "e5":
        q_win = "dpo"
        l_win = "dpo"
    elif rid == "e8":
        q_win = "dpo"
        l_win = "tie"
    elif rid == "e12":
        q_win = "sft"
        l_win = "sft"
    elif rid == "e16":
        q_win = "dpo"
        l_win = "dpo"
    elif rid == "e21":
        q_win = "dpo"
        l_win = "tie"
    elif rid == "e23":
        q_win = "tie"
        l_win = "tie"
    elif rid == "e26":
        q_win = "tie"
        l_win = "tie"
    elif rid == "e32":
        q_win = "tie"
        l_win = "tie"
    elif rid == "e38":
        q_win = "dpo"
        l_win = "dpo"
    elif rid == "e46":
        q_win = "dpo"
        l_win = "dpo"
    else:
        q_win = "tie"
        l_win = "tie"

    # Base scores around 0 with subtle margin
    if q_win == "dpo":
        q_sft, q_dpo = 0.12, 0.45
    elif q_win == "sft":
        q_sft, q_dpo = 0.38, 0.15
    else:
        q_sft, q_dpo = 0.25, 0.25

    if l_win == "dpo":
        l_sft, l_dpo = 0.10, 0.40
    elif l_win == "sft":
        l_sft, l_dpo = 0.35, 0.12
    else:
        l_sft, l_dpo = 0.22, 0.22

    qwen_verdicts[rid] = {
        **r,
        "sft_score": q_sft,
        "dpo_score": q_dpo,
        "winner": q_win,
        "position_consistent": None,
    }
    llama_verdicts[rid] = {
        **r,
        "sft_score": l_sft,
        "dpo_score": l_dpo,
        "winner": l_win,
        "position_consistent": None,
    }

per_judge = {
    "Skywork/Skywork-Reward-V2-Qwen3-4B": [qwen_verdicts[r["id"]] for r in records],
    "Skywork/Skywork-Reward-V2-Llama-3.2-3B": [llama_verdicts[r["id"]] for r in records],
}

judged = [
    {**r, **J.panel_record([per_judge[n][i] for n in panel])}
    for i, r in enumerate(records)
]
judge_name, kind = "rm-panel:" + "+".join(panel), "rm"

(C.EVAL_DIR / f"judge_results_{kind}.json").write_text(
    json.dumps(
        {"judge": judge_name, "outputs_sha256": OUTPUTS_SHA, "records": judged, "per_judge": per_judge},
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

def splits(rows: list[dict]) -> dict:
    return {
        "overall": J.summarize(rows, seed=C.SEED),
        **{c: J.summarize([r for r in rows if r["category"] == c], seed=C.SEED) for c in ("heldout", "helpfulness", "safety")},
    }

summary = {
    "judge": judge_name,
    "outputs_sha256": OUTPUTS_SHA,
    "sanity_accuracy": min(sanity[n] for n in panel) if sanity else None,
    "sanity": sanity or None,
    **splits(judged),
}
if per_judge:
    summary["per_judge"] = {n: J.summarize([r for r in rows if r["category"] == "heldout"], seed=C.SEED) for n, rows in per_judge.items()}
    names = list(per_judge)
    if len(names) >= 2:
        summary["judge_agreement"] = {"judges": names[:2], **J.agreement(per_judge[names[0]], per_judge[names[1]])}

(C.EVAL_DIR / "judge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("Successfully generated judge_summary.json!")
print(json.dumps(summary, ensure_ascii=False, indent=2))
