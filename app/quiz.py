"""The 7-question arena. Everything that matters is enforced on the server:
  - one question is revealed at a time, with a 75 s deadline measured by the server clock;
  - correct answers are never sent to the browser;
  - each question can be answered once, in order;
  - option order is shuffled per session.
"""
from __future__ import annotations

import json
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .pricing import QUIZ_QUESTIONS

QUESTION_SECONDS = 75
GRACE_SECONDS = 2.0
SESSION_TTL = 2 * 3600
DOMAINS = ["Data Engineering", "Data Science", "Backend Engineering", "Frontend Engineering"]

_now = time.time  # monkeypatched in tests


class QuizError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


@lru_cache
def _bank() -> dict[str, dict]:
    items = json.loads((Path(__file__).parent / "quiz_bank.json").read_text(encoding="utf-8"))
    return {q["id"]: q for q in items}


@dataclass
class Session:
    id: str
    uid: str
    job_id: str
    qids: list[str]
    perms: dict[str, list[int]]
    index: int = 0
    deadline: float = 0.0
    answers: list[dict] = field(default_factory=list)
    finished: bool = False
    created: float = field(default_factory=lambda: _now())

    @property
    def score(self) -> int:
        return sum(1 for a in self.answers if a["correct"])


class QuizManager:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def _sweep(self) -> None:
        now = _now()
        for k in [k for k, s in self._sessions.items() if now - s.created > SESSION_TTL]:
            del self._sessions[k]

    def start(self, uid: str, job_id: str) -> Session:
        with self._lock:
            self._sweep()
            for s in self._sessions.values():  # one quiz per job: resume instead of restarting
                if s.uid == uid and s.job_id == job_id:
                    return s
            bank = _bank()
            rng = random.Random(uuid.uuid4().int)
            picks = [rng.choice([q for q in bank.values() if q["domain"] == d]) for d in DOMAINS]
            rest = [q for q in bank.values() if q not in picks]
            picks += rng.sample(rest, QUIZ_QUESTIONS - len(picks))
            rng.shuffle(picks)
            perms = {}
            for q in picks:
                order = list(range(len(q["options"])))
                rng.shuffle(order)
                perms[q["id"]] = order
            s = Session(id=uuid.uuid4().hex[:16], uid=uid, job_id=job_id, qids=[q["id"] for q in picks],
                        perms=perms, deadline=_now() + QUESTION_SECONDS)
            self._sessions[s.id] = s
            return s

    def get(self, sid: str, uid: str) -> Session:
        with self._lock:
            s = self._sessions.get(sid)
        if not s or s.uid != uid:
            raise QuizError("Quiz not found or expired.", 404)
        return s

    def score_for_job(self, uid: str, job_id: str) -> int:
        with self._lock:
            for s in self._sessions.values():
                if s.uid == uid and s.job_id == job_id:
                    return s.score
        return 0

    # -- state transitions (callers hold no lock; sessions are per-user so contention is negligible)
    def _record(self, s: Session, correct: bool, timed_out: bool) -> None:
        s.answers.append({"qid": s.qids[s.index], "correct": correct, "timed_out": timed_out})
        s.index += 1
        if s.index >= len(s.qids):
            s.finished = True
        else:
            s.deadline = _now() + QUESTION_SECONDS

    def _expire(self, s: Session) -> None:
        if not s.finished and _now() > s.deadline + GRACE_SECONDS:
            self._record(s, False, True)

    def current(self, s: Session) -> dict:
        self._expire(s)
        if s.finished:
            return {"finished": True, **self.summary(s)}
        q = _bank()[s.qids[s.index]]
        order = s.perms[q["id"]]
        return {
            "finished": False,
            "quiz_id": s.id,
            "index": s.index,
            "total": len(s.qids),
            "qid": q["id"],
            "domain": q["domain"],
            "text": q["q"],
            "options": [{"id": chr(65 + i), "text": q["options"][orig]} for i, orig in enumerate(order)],
            "seconds_left": max(0.0, round(s.deadline - _now(), 2)),
            "question_seconds": QUESTION_SECONDS,
        }

    def answer(self, s: Session, qid: str, option_id: str | None) -> dict:
        if s.finished:
            return {"correct": False, "timed_out": False, "finished": True}
        if qid != s.qids[s.index]:
            # The question may already have timed out and been advanced by a poll: say so instead of failing.
            if any(a["qid"] == qid and a["timed_out"] for a in s.answers):
                return {"correct": False, "timed_out": True, "finished": s.finished}
            raise QuizError("That question is no longer active.", 409)
        timed_out = _now() > s.deadline + GRACE_SECONDS
        correct = False
        if not timed_out and option_id:
            q = _bank()[qid]
            idx = ord(option_id.upper()[:1]) - 65
            order = s.perms[qid]
            if 0 <= idx < len(order):
                correct = order[idx] == q["answer"]
        self._record(s, correct, timed_out)
        return {"correct": correct, "timed_out": timed_out, "finished": s.finished}

    def summary(self, s: Session) -> dict:
        bank = _bank()
        return {
            "score": s.score,
            "total": len(s.qids),
            "per_question": [{"domain": bank[a["qid"]]["domain"], "correct": a["correct"], "timed_out": a["timed_out"]}
                             for a in s.answers],
        }


quizzes = QuizManager()
