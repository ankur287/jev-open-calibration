"""Open reimplementation of Jev's System One decision interface.

The contract (from TypeSafe's public docs): a *state* (the thing to judge)
plus one or more *typed questions* goes in; typed answers with probabilities
come out. No text generation, nothing to parse.

Three question types:
  - Choice: fixed set of options, returns full probability distribution.
  - Score:  ordered levels, returns the expectation (a position between levels).
  - Noul:   truth question, returns P(true).

Backends implement `answer(state, questions)`. Two are planned:
  - SklearnBackend: any probabilistic sklearn classifier (open reconstruction).
  - JevApiBackend:  the real Jev model via TypeSafe's API (see exp04_jev_live.py).
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------- questions

@dataclass
class ChoiceQuestion:
    name: str
    instructions: str
    criteria: dict  # option -> description string


@dataclass
class ScoreQuestion:
    name: str
    instructions: str
    levels: list  # 2..10 ordered descriptions, low -> high


@dataclass
class NoulQuestion:
    name: str
    instructions: str


# ---------------------------------------------------------------- answers

def _concentration_confidence(probs):
    """Our analog of Jev's confidence: how concentrated the distribution is.

    1.0 = all mass on one option, 0.0 = uniform. This is deliberately *not*
    the top probability: it measures the shape of the belief, which is what
    makes abstention a first-class, routable outcome.
    """
    import numpy as np
    p = np.asarray(probs, dtype=float)
    p = p / p.sum()
    n = len(p)
    if n <= 1:
        return 1.0
    entropy = -(p * np.log(p + 1e-12)).sum()
    return float(1.0 - entropy / np.log(n))


@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: dict
    confidence: float


@dataclass
class ScoreAnswer:
    score: float  # expectation over level indices: may land between levels
    probabilities: dict
    legend: dict
    confidence: float


@dataclass
class NoulAnswer:
    p_true: float
    confidence: float


# ---------------------------------------------------------------- backends

class Backend:
    def answer(self, state, questions):
        """state: str (or dict); questions: list of question objects.
        Returns dict name -> answer object."""
        raise NotImplementedError


class SklearnBackend(Backend):
    """Wraps a fitted probabilistic sklearn classifier as a Choice backend.

    The classifier must expose predict_proba. Class labels map onto the
    Choice options in order.
    """

    def __init__(self, clf, vectorizer=None):
        self.clf = clf
        self.vectorizer = vectorizer

    def _features(self, state):
        X = [state] if isinstance(state, str) else [state]
        return self.vectorizer.transform(X) if self.vectorizer else X

    def answer(self, state, questions):
        X = self._features(state)
        proba = self.clf.predict_proba(X)[0]
        classes = list(self.clf.classes_)
        out = {}
        for q in questions:
            if not isinstance(q, ChoiceQuestion):
                raise ValueError(f"SklearnBackend only supports Choice, got {type(q).__name__}")
            options = list(q.criteria.keys())
            probs = {opt: float(proba[classes.index(opt)])
                     for opt in options if opt in classes}
            total = sum(probs.values()) or 1.0
            probs = {k: v / total for k, v in probs.items()}
            top = max(probs, key=probs.get)
            out[q.name] = ChoiceAnswer(choice=top, probabilities=probs,
                                       confidence=_concentration_confidence(list(probs.values())))
        return out
