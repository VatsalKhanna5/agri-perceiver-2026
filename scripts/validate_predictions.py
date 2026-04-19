#!/usr/bin/env python3
"""Validate predictions.jsonl quality — JSON validity, schema, distributions."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/predictions.jsonl"

fields = ["diagnosis", "type", "severity", "confidence", "symptoms", "reasoning", "recommended_actions"]
valid = 0
invalid = 0
truncated = 0
errors = 0
has_all_fields = 0
sevs = []
confs = []
types = {}
times = []
i = 0

with open(path) as f:
    for i, line in enumerate(f, 1):
        rec = json.loads(line)
        if "error" in rec:
            errors += 1
            continue
        times.append(rec.get("inference_time_s", 0))
        try:
            out = json.loads(rec["output"])
            valid += 1
            if all(k in out for k in fields):
                has_all_fields += 1
            sevs.append(out.get("severity", 0))
            confs.append(out.get("confidence", 0))
            t = out.get("type", "?")
            types[t] = types.get(t, 0) + 1
        except json.JSONDecodeError:
            invalid += 1
            if not rec["output"].rstrip().endswith("}"):
                truncated += 1

if i == 0:
    print("No predictions found!")
    sys.exit(1)

print(f"Total predictions: {i}")
print(f"Valid JSON: {valid} ({100*valid/i:.1f}%)")
print(f"Schema-compliant (7 fields): {has_all_fields} ({100*has_all_fields/i:.1f}%)")
print(f"Invalid JSON: {invalid} (truncated: {truncated})")
print(f"Error records: {errors}")
if sevs:
    print(f"Severity: [{min(sevs):.2f}, {max(sevs):.2f}], mean={sum(sevs)/len(sevs):.3f}")
    print(f"Confidence: [{min(confs):.2f}, {max(confs):.2f}], mean={sum(confs)/len(confs):.3f}")
if times:
    print(f"Inference time: mean={sum(times)/len(times):.2f}s, min={min(times):.2f}s, max={max(times):.2f}s")
print(f"Type distribution: {dict(sorted(types.items(), key=lambda x: -x[1]))}")
