import pytest

from app import quiz as quiz_mod
from app.quiz import QUESTION_SECONDS, QuizError, quizzes, _bank


@pytest.fixture
def clock(monkeypatch):
    t = {"now": 10_000.0}
    monkeypatch.setattr(quiz_mod, "_now", lambda: t["now"])
    return t


def correct_letter(s, payload):
    q = _bank()[payload["qid"]]
    order = s.perms[payload["qid"]]
    return chr(65 + order.index(q["answer"]))


def test_seven_questions_covering_all_domains(clock):
    s = quizzes.start("u1", "job1")
    domains = {_bank()[q]["domain"] for q in s.qids}
    assert len(s.qids) == 7 and len(set(s.qids)) == 7
    assert domains == set(quiz_mod.DOMAINS)


def test_payload_never_contains_the_answer(clock):
    s = quizzes.start("u1", "job1")
    p = quizzes.current(s)
    assert "answer" not in str(p).lower().replace("answered", "")
    assert p["seconds_left"] == QUESTION_SECONDS and len(p["options"]) == 4


def test_correct_answers_score_and_finish(clock):
    s = quizzes.start("u1", "job1")
    for _ in range(7):
        p = quizzes.current(s)
        out = quizzes.answer(s, p["qid"], correct_letter(s, p))
        assert out["correct"]
    assert s.finished and s.score == 7
    assert quizzes.current(s)["finished"] is True
    assert quizzes.score_for_job("u1", "job1") == 7


def test_wrong_answer_scores_zero(clock):
    s = quizzes.start("u1", "job1")
    p = quizzes.current(s)
    wrong = next(o["id"] for o in p["options"] if o["id"] != correct_letter(s, p))
    assert quizzes.answer(s, p["qid"], wrong)["correct"] is False
    assert s.score == 0 and s.index == 1


def test_question_times_out_after_75_seconds(clock):
    s = quizzes.start("u1", "job1")
    p = quizzes.current(s)
    clock["now"] += QUESTION_SECONDS + 5
    out = quizzes.answer(s, p["qid"], correct_letter(s, p))  # right answer, but too late
    assert out["timed_out"] and not out["correct"]
    assert s.index == 1


def test_polling_after_expiry_advances_and_resets_timer(clock):
    s = quizzes.start("u1", "job1")
    first = quizzes.current(s)["qid"]
    clock["now"] += QUESTION_SECONDS + 3
    nxt = quizzes.current(s)
    assert nxt["qid"] != first and nxt["index"] == 1
    assert nxt["seconds_left"] == QUESTION_SECONDS
    assert s.answers[0]["timed_out"]


def test_cannot_answer_a_stale_question(clock):
    s = quizzes.start("u1", "job1")
    p = quizzes.current(s)
    quizzes.answer(s, p["qid"], "A")
    with pytest.raises(QuizError):
        quizzes.answer(s, p["qid"], "A")


def test_one_quiz_per_job_and_isolated_per_user(clock):
    a = quizzes.start("u1", "job1")
    assert quizzes.start("u1", "job1") is a
    with pytest.raises(QuizError):
        quizzes.get(a.id, "someone-else")
