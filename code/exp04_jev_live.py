#!/usr/bin/env python3
"""exp04 — live Jev experiments via TypeSafe's direct Decisions API.

Part A: calibration + framing consistency on the banking7 test slice (280 items).
        One call per item: 1 Choice question + 7 Noul questions (evaluated in
        isolation per TypeSafe docs). Metrics: accuracy, ECE, Brier of Choice;
        P_noul vs P_choice agreement; contradiction rate at act@0.9; latency.
Part B: none-of-the-above trap on 100 out-of-schema items (sampled from the 70
        excluded intents, random_state=7). Measure false auto-act at t=0.9.

Usage: TYPESAFE_API_KEY=... python code/exp04_jev_live.py --probe   # 1 call, print raw response
       TYPESAFE_API_KEY=... python code/exp04_jev_live.py --full    # full run, background recommended

The API key is read from the environment and never written to disk.
"""
import json
import math
import os
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

INTENTS = {
    "card_payment_fee_charged":
        "The customer was charged a fee for paying with their card and is asking about or disputing it.",
    "direct_debit_payment_not_recognised":
        "The customer does not recognise a direct debit payment on their statement and believes they did not make it.",
    "balance_not_updated_after_cheque_or_cash_deposit":
        "The customer deposited a cheque or cash and the account balance has not updated; they are asking where the money is.",
    "wrong_amount_of_cash_received":
        "The customer withdrew or received cash but got the wrong amount, for example only a partial amount.",
    "cash_withdrawal_charge":
        "The customer was charged a fee for withdrawing cash, for example at an ATM, and is asking why.",
    "transaction_charged_twice":
        "The same transaction appears twice on the statement; the customer was charged a duplicate amount.",
    "declined_cash_withdrawal":
        "The customer's cash withdrawal was declined and they want to know why.",
}
INTENT_LIST = list(INTENTS.keys())

CHOICE_Q = {
    "type": "choice",
    "instructions": "Classify this customer support message into exactly one of the listed intents.",
    "criteria": INTENTS,
}


def noul_q(intent):
    return {
        "type": "noul",
        "instructions": "This customer message matches the following situation: " + INTENTS[intent],
    }


def call_jev(state, questions, api_key, timeout=90, retries=3):
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
        except Exception as e:  # noqa: BLE001 - retry on transient failures
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Jev call failed after {retries} attempts: {last_err}")


def parse_answers(payload):
    """Return the answers dict regardless of top-level envelope shape."""
    if isinstance(payload, dict):
        for key in ("answers", "results", "data"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        # maybe the payload itself is the answers dict
        if any(isinstance(v, dict) and ("choice" in v or "noul" in v) for v in payload.values()):
            return payload
    raise ValueError(f"unexpected response shape: {str(payload)[:300]}")


def load_banking7():
    import pandas as pd
    te = pd.read_csv(os.path.join(BASE, "data", "banking77_test.csv"))
    te = te[te["category"].isin(INTENT_LIST)].reset_index(drop=True)
    return te


def ece_score(y_true, p_true, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for yt, p in zip(y_true, p_true):
        b = min(int(p * n_bins), n_bins - 1)
        bins[b].append((yt, p))
    ece, n = 0.0, len(y_true)
    for b in bins:
        if not b:
            continue
        acc = sum(yt for yt, _ in b) / len(b)
        conf = sum(p for _, p in b) / len(b)
        ece += len(b) / n * abs(acc - conf)
    return ece


def main():
    api_key = os.environ.get("TYPESAFE_API_KEY", "")
    if not api_key:
        sys.exit("TYPESAFE_API_KEY not set")
    mode = sys.argv[1] if len(sys.argv) > 1 else "--probe"

    if mode == "--probe":
        payload, dt = call_jev(
            "Why was I charged a fee when I paid with my card?",
            {"intent": CHOICE_Q, "noul_fee": noul_q("card_payment_fee_charged")},
            api_key)
        print(f"latency: {dt:.2f}s")
        print(json.dumps(payload, indent=1)[:3000])
        return

    import pandas as pd
    import threading
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
    raw_path = os.path.join(BASE, "data", "exp04_raw.jsonl")
    raw_f = open(raw_path, "w")
    lock = threading.Lock()
    progress = {"done": 0}

    def run_call(tag, idx, state, questions):
        payload, dt = call_jev(state, questions, api_key)
        with lock:
            raw_f.write(json.dumps({"part": tag, "i": int(idx), "latency": dt,
                                    "payload": payload}) + "\n")
            progress["done"] += 1
            if progress["done"] % 40 == 0:
                print(f"  {progress['done']} calls done", flush=True)
        return idx, payload, dt

    # ---------- Part A ----------
    te = load_banking7()
    n = len(te)
    print(f"Part A: {n} banking7 items, 8 questions/call", flush=True)
    jobs_a = []
    for i, row in te.iterrows():
        questions = {"intent": CHOICE_Q}
        for intent in INTENT_LIST:
            questions[f"noul_{intent}"] = noul_q(intent)
        jobs_a.append(("A", i, row["text"], row["category"], questions))

    def do_a(job):
        tag, i, text, category, questions = job
        i, payload, dt = run_call(tag, i, text, questions)
        ans = parse_answers(payload)
        ch = ans["intent"]
        probs = ch["probabilities"]
        pred = ch["choice"]
        p_pred = probs[pred]
        diffs_i, pairs_i = [], []
        for intent in INTENT_LIST:
            pn = ans[f"noul_{intent}"]["noul"]
            pc = probs[intent]
            diffs_i.append(abs(pn - pc))
            pairs_i.append((pn, pc))
        pn_pred = ans[f"noul_{pred}"]["noul"]
        contra = (p_pred >= 0.9) != (pn_pred >= 0.9)
        return {"correct": pred == category, "p": p_pred,
                "diffs": diffs_i, "pairs": pairs_i, "dt": dt,
                "contra": contra,
                "choice_act_only": (p_pred >= 0.9) and not (pn_pred >= 0.9),
                "noul_act_only": (pn_pred >= 0.9) and not (p_pred >= 0.9)}

    with ThreadPoolExecutor(max_workers=8) as ex:
        res_a = list(ex.map(do_a, jobs_a))

    correct = [r["correct"] for r in res_a]
    top_p = [r["p"] for r in res_a]
    latencies = [r["dt"] for r in res_a]
    diffs = [d for r in res_a for d in r["diffs"]]
    pairs = [p for r in res_a for p in r["pairs"]]
    contradictions = sum(r["contra"] for r in res_a)
    acts_choice_only = sum(r["choice_act_only"] for r in res_a)
    acts_noul_only = sum(r["noul_act_only"] for r in res_a)

    acc = sum(correct) / n
    ece = ece_score([1 if c else 0 for c in correct], top_p)
    brier = sum((p - (1 if c else 0)) ** 2 for p, c in zip(top_p, correct)) / n
    mean_abs_diff = sum(diffs) / len(diffs)
    import math as _m
    mpn = sum(p for p, _ in pairs) / len(pairs)
    mpc = sum(p for _, p in pairs) / len(pairs)
    cov = sum((a - mpn) * (b - mpc) for a, b in pairs) / len(pairs)
    va = sum((a - mpn) ** 2 for a, _ in pairs) / len(pairs)
    vb = sum((b - mpc) ** 2 for _, b in pairs) / len(pairs)
    pearson = cov / _m.sqrt(va * vb)
    part_a = {
        "n": n, "accuracy": acc, "ece": ece, "brier": brier,
        "latency_s": {"mean": sum(latencies) / n,
                      "p50": sorted(latencies)[n // 2],
                      "p95": sorted(latencies)[int(n * 0.95)],
                      "max": max(latencies)},
        "framing": {"mean_abs_diff": mean_abs_diff, "pearson_r": pearson,
                    "contradiction_rate": contradictions / n,
                    "acts_choice_only": acts_choice_only,
                    "acts_noul_only": acts_noul_only},
    }
    print("\nPart A results:", json.dumps(part_a, indent=1), flush=True)

    # ---------- Part B ----------
    full = pd.read_csv(os.path.join(BASE, "data", "banking77_test.csv"))
    oos = full[~full["category"].isin(INTENT_LIST)].sample(
        100, random_state=7).reset_index(drop=True)
    print(f"\nPart B: {len(oos)} out-of-schema items", flush=True)
    def do_b(job):
        tag, i, text = job
        i, payload, dt = run_call(tag, i, text, {"intent": CHOICE_Q})
        ch = parse_answers(payload)["intent"]
        return ch["probabilities"][ch["choice"]], dt

    jobs_b = [("B", i, row["text"]) for i, row in oos.iterrows()]
    with ThreadPoolExecutor(max_workers=8) as ex:
        res_b = list(ex.map(do_b, jobs_b))
    b_top = [p for p, _ in res_b]
    b_lat = [dt for _, dt in res_b]
    part_b = {
        "n": len(oos),
        "mean_top_p": sum(b_top) / len(b_top),
        "false_auto_act_rate_at_0.9": sum(p >= 0.9 for p in b_top) / len(b_top),
        "latency_s": {"mean": sum(b_lat) / len(b_lat),
                      "p50": sorted(b_lat)[len(b_lat) // 2]},
    }
    print("\nPart B results:", json.dumps(part_b, indent=1), flush=True)
    raw_f.close()

    with open(os.path.join(BASE, "data", "exp04_results.json"), "w") as f:
        json.dump({"partA": part_a, "partB": part_b}, f, indent=2)
    print(f"\nresults -> data/exp04_results.json, raw -> {raw_path}")


if __name__ == "__main__":
    main()
