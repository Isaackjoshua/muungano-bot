# app/main.py
#
# WhatsApp bot for "Elimu ya Muungano Ubunifu Challenge" — v2.
#
# Three modes, chosen from a main menu, per phone number:
#   1) Uliza swali  -> free-text Q&A, grounded ONLY in knowledge_base.md
#   2) Quiz         -> 5-question multiple choice quiz (same as v1)
#   3) Je, Wajua?   -> cycles through short facts, one per request
#
# Design choice worth stating explicitly: this is NOT a vector-search RAG
# pipeline. knowledge_base.md is small enough (a few thousand words) to
# pass in full as the system prompt's grounding context on every request.
# That's simpler and more reliable than standing up embeddings/a vector
# DB for a corpus this size — don't add that complexity unless the corpus
# grows to the point it stops fitting comfortably in context.
#
# Requires a GEMINI_API_KEY environment variable at deploy time (free
# tier, no card required — see README.md). Model: gemini-2.5-flash,
# overridable via GEMINI_MODEL env var.

import os
import re
import sqlite3
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from fastapi import FastAPI, Form
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse

# Optional: load a local .env for development convenience. Production
# deploys set real environment variables, so this is non-fatal if either
# python-dotenv or the .env file is absent.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from app.facts import FACTS
from app.quiz_data import QUESTIONS

DB_PATH = Path(__file__).parent / "quiz_state.db"
KB_PATH = Path(__file__).parent / "knowledge_base.md"
GREETINGS = {"start", "hi", "hujambo", "habari", "mambo", "anza"}
# Configurable via env var so a rate-limit or deprecation issue is a config
# change, not a code change. gemini-2.5-flash is the current documented
# stable model ID as of mid-2026 — verify this is still current in
# Google AI Studio before deploying; free-tier RPM/RPD figures for Flash
# have changed more than once recently, so check your live quota panel
# rather than trusting any cached number, including this comment.
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
# Raised from 400 after a live WhatsApp test truncated a real answer
# mid-word: a multi-point reply started but ran out of budget. 800 lets a
# normal short answer finish comfortably while still fitting a WhatsApp
# message; the prompt (rule 4) keeps answers brief so we rarely approach it.
MAX_TOKENS = 800

MENU_TEXT = (
    "Karibu kwenye *Elimu ya Muungano*! 🇹🇿\n\n"
    "Chagua namba:\n"
    "1) Uliza swali kuhusu Muungano\n"
    "2) Fanya Quiz\n"
    "3) Je, Wajua? (Ukweli wa haraka)\n\n"
    "Andika 'MENU' wakati wowote kurudi hapa."
)

SYSTEM_PROMPT_TEMPLATE = """Wewe ni msaidizi wa maswali kuhusu Muungano wa \
Tanganyika na Zanzibar, kwa ajili ya shindano la "Elimu ya Muungano \
Ubunifu Challenge".

MAELEKEZO MAKALI — FUATA KWA USAHIHI:
1. Jibu maswali TU kwa kutumia taarifa zilizomo kwenye HATI YA MAREJEO \
hapa chini. Usitumie taarifa nyingine yoyote nje ya hati hii, hata kama \
unaijua kwa uhakika.
2. Ikiwa swali haliwezi kujibiwa kwa taarifa iliyomo humu, sema kwa \
heshima kwamba huna uhakika na taarifa hiyo, na mshauri mtumiaji auliza \
mwalimu wake au atembelee vyanzo rasmi vya Serikali. USIBUNI jibu.
3. Ikiwa swali linahusu mada zenye utata wa kisiasa (kwa mfano: \
malalamiko kuhusu muundo wa Muungano, hisia za baadhi ya Wazanzibari, \
au siasa za sasa), USICHUKUE upande wowote. Sema tu kwamba ni mada \
inayojadiliwa na wataalamu na wanasiasa mbalimbali, bila kutoa maoni \
yako mwenyewe, na mshauri kutafuta vyanzo rasmi zaidi.
4. Jibu kwa Kiswahili fasaha, kwa ufupi (sentensi 2-5) katika aya moja \
fupi inayofaa kwa ujumbe wa WhatsApp. EPUKA orodha ndefu za vidoti; \
kamilisha jibu lako kila wakati — usiache sentensi katikati.
5. Usijibu maswali yasiyohusiana kabisa na Muungano wa Tanzania — \
mkumbushe mtumiaji kwa heshima kwamba wewe ni bot ya Muungano pekee.

HATI YA MAREJEO:
{knowledge_base}
"""

app = FastAPI()

_kb_cache: str | None = None
_client: genai.Client | None = None


def load_knowledge_base() -> str:
    global _kb_cache
    if _kb_cache is None:
        _kb_cache = KB_PATH.read_text(encoding="utf-8")
    return _kb_cache


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY haijawekwa kwenye mazingira.")
        _client = genai.Client(api_key=api_key)
    return _client


def ask_llm(client: genai.Client, question: str) -> str:
    system = SYSTEM_PROMPT_TEMPLATE.format(knowledge_base=load_knowledge_base())
    resp = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=MAX_TOKENS,
        ),
    )
    return (resp.text or "").strip() or "Samahani, sikuweza kupata jibu. Jaribu tena."


# --- persistence -----------------------------------------------------------


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_state (
            phone TEXT PRIMARY KEY,
            mode TEXT NOT NULL DEFAULT 'menu',
            current_question INTEGER NOT NULL DEFAULT 0,
            score INTEGER NOT NULL DEFAULT 0,
            fact_index INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()
    conn.close()


init_db()  # also run at import time — some deployment setups don't fire
           # the ASGI startup event reliably; don't rely on it alone.


def get_state(phone: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT mode, current_question, score, fact_index FROM user_state WHERE phone = ?",
        (phone,),
    ).fetchone()
    if row is None:
        conn.execute("INSERT INTO user_state (phone) VALUES (?)", (phone,))
        conn.commit()
        conn.close()
        return {"mode": "menu", "current_question": 0, "score": 0, "fact_index": 0}
    conn.close()
    return {"mode": row[0], "current_question": row[1], "score": row[2], "fact_index": row[3]}


def set_state(phone: str, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [phone]
    conn = sqlite3.connect(DB_PATH)
    conn.execute(f"UPDATE user_state SET {cols} WHERE phone = ?", values)
    conn.commit()
    conn.close()


# --- quiz mode ---------------------------------------------------------


def format_question(idx: int) -> str:
    q = QUESTIONS[idx]
    options = "\n".join(f"{k}) {v}" for k, v in q["options"].items())
    return f"Swali {idx + 1}/{len(QUESTIONS)}:\n{q['text']}\n\n{options}"


def handle_quiz_turn(phone: str, state: dict, raw: str) -> str:
    q_idx = state["current_question"]
    score = state["score"]

    if q_idx >= len(QUESTIONS):
        set_state(phone, mode="menu")
        return f"Umeshamaliza quiz! Alama zako: {score}/{len(QUESTIONS)}.\n\n{MENU_TEXT}"

    question = QUESTIONS[q_idx]
    # STRICT match required: the whole message must be a single option
    # letter, optionally with '.' or ')'. Do NOT take the first character
    # of arbitrary text — "blah".upper()[:1] == "B" was a real bug caught
    # in testing; garbage input must be rejected, not silently graded.
    match = re.fullmatch(r"\s*([A-Da-d])[\.\)]?\s*", raw)
    user_answer = match.group(1).upper() if match else None

    if user_answer is None or user_answer not in question["options"]:
        return "Samahani, sikuelewa. Tafadhali jibu kwa A, B, C au D.\n\n" + format_question(q_idx)

    correct = user_answer == question["correct"]
    if correct:
        score += 1
        feedback = "Sahihi! ✅"
    else:
        right = question["correct"]
        feedback = f"Samahani, si sahihi. Jibu sahihi ni {right}) {question['options'][right]}."

    q_idx += 1

    if q_idx >= len(QUESTIONS):
        set_state(phone, mode="menu", current_question=q_idx, score=score)
        return f"{feedback}\n\nUmemaliza quiz! Alama zako: {score}/{len(QUESTIONS)}.\n\n{MENU_TEXT}"

    set_state(phone, current_question=q_idx, score=score)
    return f"{feedback}\n\n{format_question(q_idx)}"


# --- webhook -----------------------------------------------------------


def _xml(resp: MessagingResponse) -> Response:
    return Response(content=str(resp), media_type="application/xml")


@app.post("/webhook")
async def whatsapp_webhook(Body: str = Form(...), From: str = Form(...)):
    phone = From
    raw = Body.strip()
    lowered = raw.lower()

    resp = MessagingResponse()
    msg = resp.message()

    # Global override: MENU always returns to the main menu, regardless
    # of what mode the user was in.
    if lowered == "menu":
        set_state(phone, mode="menu")
        msg.body(MENU_TEXT)
        return _xml(resp)

    state = get_state(phone)

    if state["mode"] == "menu":
        if lowered in GREETINGS:
            msg.body(MENU_TEXT)
            return _xml(resp)

        choice = raw.strip()
        if choice == "1":
            set_state(phone, mode="ask")
            msg.body("Uliza swali lolote kuhusu Muungano, kwa Kiswahili.\n(Andika MENU kurudi.)")
        elif choice == "2":
            set_state(phone, mode="quiz", current_question=0, score=0)
            msg.body("Karibu kwenye Quiz! Jibu kwa A, B, C au D.\n\n" + format_question(0))
        elif choice == "3":
            idx = state["fact_index"] % len(FACTS)
            set_state(phone, fact_index=state["fact_index"] + 1)
            msg.body(
                f"💡 Je, Wajua?\n\n{FACTS[idx]}\n\n"
                "Andika 3 kwa ukweli mwingine, au MENU kurudi kwenye menyu."
            )
        else:
            msg.body("Samahani, sikuelewa.\n\n" + MENU_TEXT)
        return _xml(resp)

    if state["mode"] == "quiz":
        msg.body(handle_quiz_turn(phone, state, raw))
        return _xml(resp)

    if state["mode"] == "ask":
        try:
            answer = ask_llm(get_client(), raw)
        except RuntimeError as exc:
            answer = f"Samahani, huduma ya maswali haipo kwa sasa ({exc})"
        except genai_errors.APIError as exc:
            if getattr(exc, "code", None) == 429:
                # Expected on a free tier under bursty traffic — tell the
                # user to retry rather than a generic technical-error
                # message, since this isn't really a fault.
                answer = "Samahani, huduma ina watumiaji wengi kwa sasa. Tafadhali jaribu tena baada ya dakika chache."
            else:
                answer = "Samahani, kuna hitilafu ya kiufundi. Jaribu tena baadaye."
        except Exception as exc:  # noqa: BLE001 - last line of defense so a
            # deploy mistake (e.g. missing knowledge_base.md) never surfaces
            # as a hard 500 to a WhatsApp user; log it so it's actually
            # visible when running the server.
            print(f"[ask mode] unexpected error: {exc!r}")
            answer = "Samahani, kuna hitilafu ya kiufundi. Jaribu tena baadaye."
        msg.body(answer + "\n\n(Andika MENU kurudi.)")
        return _xml(resp)

    # Unknown mode fallback — shouldn't happen, but don't dead-end the user.
    set_state(phone, mode="menu")
    msg.body(MENU_TEXT)
    return _xml(resp)


@app.get("/health")
def health():
    return {"status": "ok"}
