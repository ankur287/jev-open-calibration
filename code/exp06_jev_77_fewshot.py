#!/usr/bin/env python3
"""exp06 — live Jev on the FULL 77-intent BANKING77 test set, few-shot criteria.

Same as exp05, but each intent's criteria text includes 3 real training examples
(closest to the class centroid in TF-IDF space) instead of the bare name.

Spectrum being mapped:
  exp05: 0-shot, terse criteria  -> ?
  exp06: 3-shot criteria         -> ?
  open logreg: 10k labeled examples -> acc 0.888, ECE 0.248
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
    import numpy as _np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    tr = pd.read_csv(os.path.join(BASE, "data", "banking77_train.csv"))
    te = pd.read_csv(os.path.join(BASE, "data", "banking77_test.csv")).reset_index(drop=True)
    intents = sorted(te["category"].unique().tolist())
    print(f"intents={len(intents)} items={len(te)}", flush=True)

    # 3 representative training examples per intent (closest to centroid)
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(tr["text"])
    criteria = {}
    for intent in intents:
        m = (tr["category"] == intent).to_numpy()
        Xm = Xtr[m]
        centroid = _np.asarray(Xm.mean(axis=0)).ravel()
        norms = _np.sqrt(_np.asarray(Xm.multiply(Xm).sum(axis=1))).ravel() + 1e-9
        sims = (Xm @ centroid) / (norms * (_np.linalg.norm(centroid) + 1e-9))
        top3 = _np.argsort(-sims)[:3]
        texts = tr[m].iloc[top3]["text"].tolist()
        ex = " / ".join(t[:120] for t in texts)
        criteria[intent] = f"{intent.replace('_', ' ')}. e.g. {ex}"
    question = {
        "type": "choice",
        "instructions": "Classify this customer support message into exactly one of the listed intents.",
        "criteria": criteria,
    }

    os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
    raw_path = os.path.join(BASE, "data", "exp06_raw.jsonl")
    done_idx = set()
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if "payload" in r:
                        done_idx.add(r["i"])
                except (json.JSONDecodeError, KeyError):
                    pass
    raw_f = open(raw_path, "a")
    lock = threading.Lock()
    done = {"n": len(done_idx)}
    print(f"resuming: {len(done_idx)} already done", flush=True)

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
        list(ex.map(do, jobs))
    raw_f.close()

    import numpy as _np2
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
    yt = _np2.array(y_true); pp = _np2.array(top_p)
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        m = (pp > lo) & (pp <= hi) if b else (pp >= lo) & (pp <= hi)
        if m.sum():
            ece += m.sum() / n * abs(yt[m].mean() - pp[m].mean())
    brier = float(((pp - yt) ** 2).mean())
    out = {"n": n, "accuracy": acc, "ece": ece, "brier": brier,
           "latency_s": {"mean": sum(lat) / n, "p50": sorted(lat)[n // 2],
                         "p95": sorted(lat)[int(n * 0.95)], "max": max(lat)}}
    print(json.dumps(out, indent=1), flush=True)
    with open(os.path.join(BASE, "data", "exp06_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"results -> data/exp06_results.json, raw -> {raw_path}")


if __name__ == "__main__":
    main()
