# tests/test_app.py
#
# Regression suite for the Muungano WhatsApp bot. Every case here mirrors a
# behaviour that was manually verified during development — several of them
# guard against specific bugs that were actually hit (see comments). Do not
# delete a test because it "looks redundant".
#
# No real Twilio or Gemini credentials are required: the Gemini client is
# mocked, and the API-key / rate-limit / missing-KB failure paths are exercised
# with monkeypatch. Each test gets an isolated on-disk SQLite DB via the autouse
# fixture below, so state never leaks between tests or between phone numbers.

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

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
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    send(client, "hi")
    send(client, "1")                    # enter ask mode
    body = send(client, "Muungano ni nini?")
    # 200 already asserted in send(); check the graceful Swahili message.
    assert "huduma ya maswali haipo" in body


# --- 10. ask mode with missing knowledge base: graceful, not crash ---------


def test_ask_mode_missing_knowledge_base(client, tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(main, "KB_PATH", tmp_path / "does_not_exist.md")
    send(client, "hi")
    send(client, "1")
    body = send(client, "Muungano ni nini?")
    assert "hitilafu ya kiufundi" in body


# --- 11. ask_llm builds the correct Gemini request (mock client) -----------


def test_ask_llm_request_shape():
    # Mimics google-genai: client.models.generate_content(...) -> obj.text
    fake = MagicMock()
    fake.models.generate_content.return_value = MagicMock(text="jibu la mfano")

    out = main.ask_llm(fake, "Muungano ulianzishwa lini?")
    assert out == "jibu la mfano"

    _, kwargs = fake.models.generate_content.call_args
    assert kwargs["model"] == main.MODEL == "gemini-2.5-flash"
    # Raw question passed through untouched.
    assert kwargs["contents"] == "Muungano ulianzishwa lini?"
    # Full knowledge base must be grounded into the system instruction.
    assert main.load_knowledge_base() in kwargs["config"].system_instruction
    # Neutrality instruction must survive.
    assert "USICHUKUE upande" in kwargs["config"].system_instruction


# --- 11b. free-tier 429 rate limit gets its own retry message --------------


def test_ask_mode_rate_limited_429(client, monkeypatch):
    # A 429 is an expected, recoverable free-tier condition — it must produce
    # the "watumiaji wengi" retry copy, NOT the generic technical-error copy.
    fake = MagicMock()
    fake.models.generate_content.side_effect = genai_errors.APIError(
        429, {"error": {"message": "rate limited", "status": "RESOURCE_EXHAUSTED"}}
    )
    # Inject the fake as the cached client so get_client() returns it without
    # needing a real key.
    monkeypatch.setattr(main, "_client", fake)

    send(client, "hi")
    send(client, "1")
    body = send(client, "Muungano ni nini?")
    assert "watumiaji wengi" in body
    assert "hitilafu ya kiufundi" not in body


# --- 12. health check ------------------------------------------------------


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --- Meta WhatsApp Cloud API channel --------------------------------------
#
# The Meta path shares ALL business logic with Twilio via process_message;
# these tests exercise the channel adapter: the GET verification handshake,
# JSON parsing, phone-key normalisation, and that the reply is dispatched via
# send_whatsapp_message (mocked — no network, no real Graph API token).

WA_ID = "255700000001"                      # Meta delivers digits only...
META_PHONE_KEY = "whatsapp:+255700000001"   # ...we normalise to Twilio's form


def _meta_text_payload(text, wa_id=WA_ID):
    """Build the nested JSON Meta sends for one inbound text message."""
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WABA_ID",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"phone_number_id": "PNID"},
                            "messages": [
                                {
                                    "from": wa_id,
                                    "id": "wamid.TEST",
                                    "timestamp": "0",
                                    "type": "text",
                                    "text": {"body": text},
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }


@pytest.fixture
def capture_send(monkeypatch):
    """Replace the Graph API sender with a mock that records (to, body)."""
    fake = MagicMock()
    monkeypatch.setattr(main, "send_whatsapp_message", fake)
    return fake


# --- M1. verification handshake succeeds with the right token --------------


def test_meta_verify_handshake_ok(client, monkeypatch):
    monkeypatch.setattr(main, "META_VERIFY_TOKEN", "secret-tok")
    resp = client.get(
        "/meta/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "secret-tok",
            "hub.challenge": "1234567890",
        },
    )
    assert resp.status_code == 200
    assert resp.text == "1234567890"     # echoed back verbatim


# --- M2. verification is rejected on a wrong token -------------------------


def test_meta_verify_handshake_bad_token(client, monkeypatch):
    monkeypatch.setattr(main, "META_VERIFY_TOKEN", "secret-tok")
    resp = client.get(
        "/meta/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "WRONG",
            "hub.challenge": "1234567890",
        },
    )
    assert resp.status_code == 403


# --- M3. inbound text runs shared logic and dispatches a reply -------------


def test_meta_inbound_text_greeting(client, capture_send):
    resp = client.post("/meta/webhook", json=_meta_text_payload("hi"))
    assert resp.status_code == 200

    capture_send.assert_called_once()
    to, body = capture_send.call_args.args
    assert to == WA_ID                    # raw wa_id used for sending
    assert "Chagua namba" in body         # the menu, from process_message


# --- M4. Meta and Twilio share state via the normalised phone key ----------


def test_meta_shares_state_with_process_message(client, capture_send):
    # Choosing "2" over Meta must start the quiz for the normalised key,
    # proving the wa_id -> whatsapp:+<id> mapping reaches the same state row.
    client.post("/meta/webhook", json=_meta_text_payload("2"))
    assert main.get_state(META_PHONE_KEY)["mode"] == "quiz"
    _, body = capture_send.call_args.args
    assert "Swali 1/5" in body


# --- M5. delivery-status callbacks are ignored (no reply sent) -------------


def test_meta_status_callback_ignored(client, capture_send):
    status_event = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WABA_ID",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"phone_number_id": "PNID"},
                            "statuses": [
                                {"id": "wamid.X", "status": "delivered"}
                            ],
                        },
                    }
                ],
            }
        ],
    }
    resp = client.post("/meta/webhook", json=status_event)
    assert resp.status_code == 200
    capture_send.assert_not_called()


# --- M6. non-text messages get a gentle "text only" nudge ------------------


def test_meta_non_text_message(client, capture_send):
    payload = _meta_text_payload("ignored")
    msg = payload["entry"][0]["changes"][0]["value"]["messages"][0]
    msg["type"] = "image"
    del msg["text"]
    msg["image"] = {"id": "media123"}

    client.post("/meta/webhook", json=payload)
    to, body = capture_send.call_args.args
    assert to == WA_ID
    assert "maandishi tu" in body         # "I understand text only"


# --- M7. send_whatsapp_message posts the correct Graph API request ---------


def test_send_whatsapp_message_shape(monkeypatch):
    monkeypatch.setattr(main, "META_ACCESS_TOKEN", "tok-abc")
    monkeypatch.setattr(main, "META_PHONE_NUMBER_ID", "PNID99")
    monkeypatch.setattr(main, "GRAPH_API_VERSION", "v22.0")

    calls = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["url"] = url
        calls["json"] = json
        calls["headers"] = headers
        return MagicMock(status_code=200, text="ok")

    monkeypatch.setattr(main.httpx, "post", fake_post)

    main.send_whatsapp_message("255700000001", "Habari!")

    assert calls["url"] == "https://graph.facebook.com/v22.0/PNID99/messages"
    assert calls["headers"]["Authorization"] == "Bearer tok-abc"
    assert calls["json"]["to"] == "255700000001"
    assert calls["json"]["text"]["body"] == "Habari!"
    assert calls["json"]["messaging_product"] == "whatsapp"


# --- M8. send is a safe no-op when Meta creds are absent -------------------


def test_send_whatsapp_message_no_creds(monkeypatch):
    monkeypatch.setattr(main, "META_ACCESS_TOKEN", None)
    monkeypatch.setattr(main, "META_PHONE_NUMBER_ID", None)
    # Must not raise and must not attempt a network call.
    boom = MagicMock(side_effect=AssertionError("should not POST"))
    monkeypatch.setattr(main.httpx, "post", boom)
    main.send_whatsapp_message("255700000001", "hi")
    boom.assert_not_called()
