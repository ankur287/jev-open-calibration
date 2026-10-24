#!/usr/bin/env python3
"""exp05 — live Jev on the FULL 77-intent BANKING77 test set (3,080 items).

Tests the "no accuracy tax" hypothesis: isotonic calibration cost 1.7pp accuracy
on 77 intents for the open models. Does Jev hold 77-intent accuracy AND calibration?

Criteria descriptions are derived tersely from intent names (disclosed in article).
One Choice question per call. 8 workers.

Caveat: BANKING77 is public; Jev may have seen it during training (contamination).
"""
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"


def call_jev(state, questions, api_key, timeout=120, retries=6):
    body = json.dumps({"model": MODEL, "state": state, "questions": questions}).encode()
    last_err = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            req = urllib.request.Request(
                ENDPOINT, data=body,
                headers={"Authorization": "Bearer " + api_key,
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.load(resp)
            return payload, time.time() - t0
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 ** attempt * 1.5)
    raise RuntimeError(f"Jev call failed after {retries} attempts: {last_err}")


def main():
    api_key = os.environ.get("TYPESAFE_API_KEY", "")
    if not api_key:
        sys.exit("TYPESAFE_API_KEY not set")
    import pandas as pd
    te = pd.read_csv(os.path.join(BASE, "data", "banking77_test.csv")).reset_index(drop=True)
    intents = sorted(te["category"].unique().tolist())
    print(f"intents={len(intents)} items={len(te)}", flush=True)
    criteria = {c: c.replace("_", " ") for c in intents}
    question = {
        "type": "choice",
        "instructions": "Classify this customer support message into exactly one of the listed intents.",
        "criteria": criteria,
    }

    os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
    raw_path = os.path.join(BASE, "data", "exp05_raw.jsonl")
    # resume: skip indices already recorded
    done_idx = set()
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            for line in f:
                try:
                    done_idx.add(json.loads(line)["i"])
                except (json.JSONDecodeError, KeyError):
                    pass
    raw_f = open(raw_path, "a")
    print(f"resuming: {len(done_idx)} already done", flush=True)
    lock = threading.Lock()
    done = {"n": len(done_idx)}

    def do(job):
        i, text = job
        try:
            payload, dt = call_jev(text, {"intent": question}, api_key)
        except RuntimeError as e:
            with lock:
                raw_f.write(json.dumps({"i": int(i), "error": str(e)[:200]}) + "\n")
                raw_f.flush()
            return None
        with lock:
            raw_f.write(json.dumps({"i": int(i), "latency": dt, "payload": payload}) + "\n")
            raw_f.flush()
            done["n"] += 1
            if done["n"] % 200 == 0:
                print(f"  {done['n']}/{len(te)}", flush=True)
        ch = payload["answers"]["intent"]
        return ch["choice"], ch["probabilities"][ch["choice"]], dt

    jobs = [(i, r["text"]) for i, r in te.iterrows() if i not in done_idx]
    print(f"submitting {len(jobs)} remaining", flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        res = [r for r in ex.map(do, jobs) if r is not None]
    raw_f.close()

    # reload all payloads in index order for metrics
    recs = {}
    with open(raw_path) as f:
        for line in f:
            r = json.loads(line)
            if "payload" in r:
                recs[r["i"]] = r
    idx = sorted(recs.keys())
    y_true, top_p, lat = [], [], []
    for i in idx:
        ch = recs[i]["payload"]["answers"]["intent"]
        top_p.append(ch["probabilities"][ch["choice"]])
        y_true.append(1 if ch["choice"] == te.loc[i, "category"] else 0)
        lat.append(recs[i]["latency"])
    n = len(y_true)
    acc = sum(y_true) / n
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        import numpy as _np
        yt = _np.array(y_true); pp = _np.array(top_p)
        m = (pp > lo) & (pp <= hi) if b else (pp >= lo) & (pp <= hi)
        if m.sum():
            ece += m.sum() / n * abs(yt[m].mean() - pp[m].mean())
    brier = sum((p - y) ** 2 for p, y in zip(top_p, y_true)) / n
    out = {"n": n, "accuracy": acc, "ece": ece, "brier": brier,
           "latency_s": {"mean": sum(lat) / n, "p50": sorted(lat)[n // 2],
                         "p95": sorted(lat)[int(n * 0.95)], "max": max(lat)}}
    print(json.dumps(out, indent=1), flush=True)
    with open(os.path.join(BASE, "data", "exp05_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"results -> data/exp05_results.json, raw -> {raw_path}")


if __name__ == "__main__":
    main()
