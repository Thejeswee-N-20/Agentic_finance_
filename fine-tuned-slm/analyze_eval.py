"""Summarise eval_finetuned.json into the numbers quoted in README.md."""
import json
import sys
from collections import Counter

ORDINAL = {
    "down by more than 5%": -6, "down by 4-5%": -5, "down by 3-4%": -4,
    "down by 2-3%": -3, "down by 1-2%": -2, "down by 0-1%": -1,
    "up by 0-1%": 1, "up by 1-2%": 2, "up by 2-3%": 3,
    "up by 3-4%": 4, "up by 4-5%": 5, "up by more than 5%": 6,
}

path = sys.argv[1] if len(sys.argv) > 1 else "eval_finetuned.json"
d = json.load(open(path))
rows = [r for r in d["rows"] if r["pred"]]
n = len(rows)

gold_up = [r for r in rows if ORDINAL[r["gold"]] > 0]
pred_up = [r for r in rows if ORDINAL[r["pred"]] > 0]

tp = sum(1 for r in rows if ORDINAL[r["gold"]] > 0 and ORDINAL[r["pred"]] > 0)
tn = sum(1 for r in rows if ORDINAL[r["gold"]] < 0 and ORDINAL[r["pred"]] < 0)
fp = sum(1 for r in rows if ORDINAL[r["gold"]] < 0 and ORDINAL[r["pred"]] > 0)
fn = sum(1 for r in rows if ORDINAL[r["gold"]] > 0 and ORDINAL[r["pred"]] < 0)

print(f"parsed            : {n}/{d['summary']['n']}  ({n / d['summary']['n']:.0%})")
print(f"exact bucket acc  : {d['summary']['exact_bucket_acc']:.1%}   (random 1/12 = 8.3%)")
print(f"within-1 bucket   : {d['summary']['within_1_bucket_acc']:.1%}")
print(f"direction acc     : {d['summary']['direction_acc']:.1%}   (coin flip = 50%)")
print(f"mean |bucket err| : {d['summary']['mean_abs_bucket_error']:.2f}")
print()
print("direction confusion (rows=gold, cols=pred)")
print(f"           pred up   pred down")
print(f"gold up    {tp:>7}   {fn:>9}")
print(f"gold down  {fp:>7}   {tn:>9}")
print()
print(f"gold up rate : {len(gold_up)}/{n} = {len(gold_up)/n:.0%}")
print(f"pred up rate : {len(pred_up)}/{n} = {len(pred_up)/n:.0%}   <- model's directional bias")
print()
maj = Counter(r["gold"] for r in rows).most_common(1)[0]
print(f"majority-class baseline: always '{maj[0]}' -> {maj[1]/n:.1%} exact")
best_dir = max(len(gold_up), n - len(gold_up)) / n
print(f"majority-direction baseline: {best_dir:.1%}")
print()
print("predicted bucket distribution:")
for b, c in Counter(r["pred"] for r in rows).most_common():
    print(f"  {b:22s} {c}")
