# tests/test_app.py
#
# Regression suite for the Muungano WhatsApp bot. Every case here mirrors a
# behaviour that was manually verified during development — several of them
# guard against specific bugs that were actually hit (see comments). Do not
# delete a test because it "looks redundant".
#
# No real Twilio or Anthropic credentials are required: the Anthropic client
# is mocked, and the API-key / missing-KB failure paths are exercised with
# monkeypatch. Each test gets an isolated on-disk SQLite DB via the autouse
# fixture below, so state never leaks between tests or between phone numbers.

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app import main

PHONE_A = "whatsapp:+255700000001"
PHONE_B = "whatsapp:+255700000002"

# Correct option keys for the 5 starter questions, in order.
CORRECT = [q["correct"] for q in main.QUESTIONS]
# A deliberately wrong key for each question (any option != the correct one).
WRONG = [
    next(k for k in q["options"] if k != q["correct"])
    for q in main.QUESTIONS
]


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Fresh DB + reset module-level caches for every test."""
    db = tmp_path / "quiz_state.db"
    monkeypatch.setattr(main, "DB_PATH", db)
    monkeypatch.setattr(main, "_client", None)
    monkeypatch.setattr(main, "_kb_cache", None)
    main.init_db()
    yield


@pytest.fixture
def client():
    # Must use the context-manager form so ASGI startup runs.
    with TestClient(main.app) as c:
        yield c


def send(client, body, frm=PHONE_A):
    """POST one WhatsApp message; return the TwiML body text."""
    resp = client.post("/webhook", data={"Body": body, "From": frm})
    assert resp.status_code == 200, resp.text
    return resp.text


# --- 1. greeting -----------------------------------------------------------


def test_greeting_shows_menu(client):
    body = send(client, "hi")
    assert "Chagua namba" in body
    assert "Fanya Quiz" in body


# --- 2. starting the quiz --------------------------------------------------


def test_choice_2_starts_quiz(client):
    send(client, "hi")
    body = send(client, "2")
    assert "Swali 1/5" in body
    assert main.QUESTIONS[0]["text"] in body


# --- 3. MENU escape hatch mid-quiz -----------------------------------------


def test_menu_override_mid_quiz(client):
    send(client, "hi")
    send(client, "2")            # start quiz
    send(client, CORRECT[0])     # answer Q1, now mid-quiz on Q2
    body = send(client, "MENU")
    assert "Chagua namba" in body
    assert main.get_state(PHONE_A)["mode"] == "menu"


# --- 4. facts do not immediately repeat ------------------------------------


def test_facts_do_not_repeat(client):
    send(client, "hi")
    first = send(client, "3")
    second = send(client, "3")
    assert main.FACTS[0][:40] in first
    assert main.FACTS[1][:40] in second
    assert first != second


# --- 5. garbage quiz input is rejected, not graded -------------------------


def test_garbage_answer_rejected(client):
    # Guards against the naive `text.upper()[:1]` bug where "blah" was
    # graded as answer "B". Must be rejected AND leave the score at 0.
    send(client, "hi")
    send(client, "2")
    body = send(client, "blah")
    assert "sikuelewa" in body
    assert "Swali 1/5" in body           # still on question 1
    state = main.get_state(PHONE_A)
    assert state["score"] == 0
    assert state["current_question"] == 0


# --- 6. full correct run ---------------------------------------------------


def test_full_correct_run_scores_5(client):
    send(client, "hi")
    send(client, "2")
    last = ""
    for key in CORRECT:
        last = send(client, key)
    assert "5/5" in last
    assert main.get_state(PHONE_A)["mode"] == "menu"


# --- 7. full wrong run -----------------------------------------------------


def test_full_wrong_run_scores_0(client):
    send(client, "hi")
    send(client, "2")
    last = ""
    for key in WRONG:
        last = send(client, key)
        assert "si sahihi" in last       # every turn gives corrective feedback
    assert "0/5" in last
    assert main.get_state(PHONE_A)["mode"] == "menu"


# --- 8. isolation between two phone numbers --------------------------------


def test_two_phones_no_state_leak(client):
    # Both start the quiz; A answers everything correctly, B everything
    # wrongly, interleaved turn-by-turn. Scores must not bleed across.
    for frm in (PHONE_A, PHONE_B):
        send(client, "hi", frm=frm)
        send(client, "2", frm=frm)

    a_last = b_last = ""
    for i in range(len(main.QUESTIONS)):
        a_last = send(client, CORRECT[i], frm=PHONE_A)
        b_last = send(client, WRONG[i], frm=PHONE_B)

    assert "5/5" in a_last
    assert "0/5" in b_last


# --- 9. ask mode with no API key: graceful, not 500 ------------------------


def test_ask_mode_missing_api_key(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    send(client, "hi")
    send(client, "1")                    # enter ask mode
    body = send(client, "Muungano ni nini?")
    # 200 already asserted in send(); check the graceful Swahili message.
    assert "huduma ya maswali haipo" in body


# --- 10. ask mode with missing knowledge base: graceful, not crash ---------


def test_ask_mode_missing_knowledge_base(client, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(main, "KB_PATH", tmp_path / "does_not_exist.md")
    send(client, "hi")
    send(client, "1")
    body = send(client, "Muungano ni nini?")
    assert "hitilafu ya kiufundi" in body


# --- 11. ask_llm builds the correct request (mock client) ------------------


def test_ask_llm_request_shape():
    class FakeBlock:
        type = "text"

        def __init__(self, text):
            self.text = text

    fake = MagicMock()
    fake.messages.create.return_value = MagicMock(content=[FakeBlock("jibu la mfano")])

    out = main.ask_llm(fake, "Muungano ulianzishwa lini?")
    assert out == "jibu la mfano"

    _, kwargs = fake.messages.create.call_args
    assert kwargs["model"] == "claude-sonnet-5"
    # Full knowledge base must be grounded into the system prompt.
    assert main.load_knowledge_base() in kwargs["system"]
    # Neutrality instruction must survive.
    assert "USICHUKUE upande" in kwargs["system"]
    # User question is passed through untouched.
    assert kwargs["messages"] == [
        {"role": "user", "content": "Muungano ulianzishwa lini?"}
    ]


# --- 12. health check ------------------------------------------------------


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
