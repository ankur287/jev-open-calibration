# jev-open-calibration

Code, data and figures for the Towards Data Science article "Jev Is API-Only. Its Best Idea Is Open."

[TypeSafe's Jev](https://docs.typesafe.ai/introduction) is a decision model with an API-only release. You send it a state and some typed questions (Choice, Score, Noul), and it returns probabilities instead of generated text. This repo rebuilds that interface on scikit-learn, measures how calibrated small open models are, and runs the same tests against the real Jev API.

## Results

| Task | Model | Accuracy | ECE |
|---|---|---|---|
| 7 intents | LogReg + temperature (open) | 0.975 | 0.022 |
| 7 intents | Jev, Choice questions | 0.975 | 0.014 (95% CI 0.008 to 0.031) |
| 77 intents | LogReg + temperature (open) | 0.888 | 0.031 |
| 77 intents | Jev, three-shot criteria | 0.859 | 0.065 |
| 77 intents | Jev, terse criteria | 0.801 | 0.083 |

All ECEs use 15 equal-width bins (`code/ece_matched_bins.py`).

Other findings:

- One temperature fit on out-of-fold logits takes ECE on the 77-intent task from 0.248 to 0.031 without changing any prediction.
- At the same 0.90 threshold, the three open backends auto-decide 0.2%, 21.1% and 78.9% of traffic.
- With no "none of these" option, out-of-schema queries get auto-acted on at 29.2% (open, temperature-scaled) and 26% (Jev).
- Jev's Noul (yes/no) outputs are underconfident on positives (ECE 0.159), while its Choice outputs are well calibrated.

## Layout

```
code/
  system_one.py          typed question/answer dataclasses + SklearnBackend
  calibration.py         ECE, reliability diagrams, TemperatureScaledLR
  exp01_calibration.py   Experiment 1: calibration of four open backends
  exp02_cascade.py       Experiment 2: confidence-gated cascade economics
  exp03_schema.py        Experiment 3: none-of-the-above trap + framing test
  exp04_jev_live.py      Experiment 4: live Jev, 7-intent slice + trap
  exp05_jev_77.py        Experiment 4: live Jev, 77 intents, terse criteria
  exp06_jev_77_fewshot.py  Experiment 4: live Jev, 77 intents, three-shot criteria
  exp01_derived.py, exp02_derived.py, bootstrap_cis.py   numbers quoted in the article
  exp03b_framing_jev_protocol.py  framing test on open models with the Jev protocol
  ece_matched_bins.py    Jev and open ECE on the same 15 bins + paired bootstrap
  example_lost_wallet.py the lost-wallet example from the article's opening
data/
  banking77_{train,test}.csv   BANKING77 (CC-BY 4.0)
  exp0*_results.json           saved results for every experiment
  exp0{4,5,6}_raw.jsonl        raw Jev API responses (6,540 calls)
figures/                       every figure in the article
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python code/exp01_calibration.py
python code/exp02_cascade.py
python code/exp03_schema.py
python code/exp01_derived.py && python code/exp02_derived.py
python code/bootstrap_cis.py    # bootstrap CIs from the saved Jev responses, offline
python code/exp03b_framing_jev_protocol.py
python code/ece_matched_bins.py
python code/example_lost_wallet.py   # the opening example, a few seconds
```

The open-model experiments take a few minutes on a laptop CPU.

The Jev experiments call the live API and need a key:

```bash
export TYPESAFE_API_KEY=...
python code/exp04_jev_live.py --probe   # one call, prints the raw response
python code/exp04_jev_live.py --full
```

The key is read from the environment and never written to disk. The full Jev run cost $0.80. You don't need a key to check the article's Jev numbers, because the raw responses are in `data/`.

## Data

BANKING77 is from Casanueva et al., [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807) (2020), released by PolyAI under CC-BY 4.0.

## License

Code is MIT licensed. See `LICENSE`.
