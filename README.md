# Muungano WhatsApp Bot — *Elimu ya Muungano Ubunifu Challenge*

A WhatsApp bot (Swahili) for a Tanzanian government-run university innovation
contest about the Union of Tanganyika and Zanzibar (*Muungano*). It runs on
FastAPI behind a Twilio WhatsApp webhook, keeps per-user state in SQLite, and
answers free-text questions with the Anthropic API — grounded strictly in a
vetted knowledge base.

## The three modes

A user picks a mode from a numbered main menu (typing `MENU` at any time
returns there):

1. **Uliza swali** — free-text Q&A. The model answers **only** from
   `app/knowledge_base.md`; it is instructed to decline politely rather than
   invent an answer, and to stay neutral on politically contested topics.
2. **Fanya Quiz** — a 5-question multiple-choice quiz (answer `A`–`D`), with
   a running score reported at the end.
3. **Je, Wajua?** — cycles through 10 short "did you know?" facts, one per
   request.

### Why not vector-search RAG?

`knowledge_base.md` is a few thousand words — small enough to pass in full as
the system prompt's grounding context on every request. That is simpler and
more reliable than standing up embeddings and a vector DB for a corpus this
size. Only revisit that if the corpus grows past comfortably fitting in
context.

### Model

Uses `claude-sonnet-5`. Sonnet 5 is the strong general-purpose Claude model —
a good fit here for careful, instruction-following Swahili answers that must
respect the "only answer from the reference document, stay neutral" rules,
without the cost of the largest model.

## Local setup

Requires Python 3.12.

```bash
cd muungano-bot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Provide your Anthropic key (either export it, or copy the example file):
cp .env.example .env      # then edit .env and paste your real key
# export ANTHROPIC_API_KEY=sk-ant-...

uvicorn app.main:app --reload
```

`.env` is loaded automatically if present (via `python-dotenv`); in production
you would set real environment variables instead. The SQLite file
`app/quiz_state.db` is created automatically on first use and is gitignored.

Health check: `GET http://127.0.0.1:8000/health` → `{"status": "ok"}`.

## Running the tests

```bash
pip install pytest
pytest -v
```

The suite mocks the Anthropic client and isolates the database per test, so it
needs **no** real Twilio or Anthropic credentials.

## Connecting the Twilio WhatsApp Sandbox

This bot targets the **Twilio WhatsApp Sandbox** (not the production WhatsApp
Business API — see limitations).

1. In the Twilio Console, go to **Messaging → Try it out → Send a WhatsApp
   message**. You'll see a sandbox number and a **join code** (e.g.
   `join <two-words>`).
2. From your phone, send that `join <two-words>` message to the sandbox number
   to opt in.
3. Under the sandbox's **Sandbox settings**, set **"When a message comes in"**
   to your webhook URL with the `/webhook` path, method **POST**:
   `https://<your-host>/webhook`.

### Local testing with ngrok

Your local server isn't reachable from Twilio directly. Tunnel it:

```bash
uvicorn app.main:app --reload      # terminal 1 (port 8000)
ngrok http 8000                    # terminal 2
```

Copy the `https://<random>.ngrok.io` URL ngrok prints and set the sandbox
webhook to `https://<random>.ngrok.io/webhook`. For the **actual submission**,
point the sandbox webhook at your deployed URL instead (e.g.
`https://your-domain/webhook`).

> **72-hour session note:** the Twilio sandbox opt-in expires after 72 hours of
> inactivity. If the bot goes quiet, re-send the `join <two-words>` code to
> reconnect before testing again.

## Deployment

Both options are scaffolded; pick one.

- **Docker:** `docker build -t muungano-bot .` then
  `docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... muungano-bot`.
  Includes a `HEALTHCHECK` against `/health`.
- **VPS + systemd:** see `deploy/muungano-bot.service`. Put the API key in
  `/etc/muungano-bot.env` (`ANTHROPIC_API_KEY=...`, `chmod 600`), install the
  app under `/opt/muungano-bot` with a virtualenv, then
  `systemctl enable --now muungano-bot`.

## Known limitations

- **Twilio Sandbox, not production.** This uses the WhatsApp Sandbox, which
  requires the `join` opt-in and expires after 72h of inactivity. Going to
  production needs an approved WhatsApp Business API sender.
- **Starter content only.** 5 quiz questions and 10 facts, deliberately
  limited to uncontroversial, well-documented historical material. Verify
  every fact against a primary/official source before final submission.
- **Contested political content is deliberately excluded** from the Q&A
  corpus. When asked about disputed aspects of the Union, the bot deflects
  neutrally (points the user to official sources / their teacher) rather than
  taking a side — an intentional scope decision documented in Section 8 of
  `app/knowledge_base.md`.
