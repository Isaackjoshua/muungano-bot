# Muungano WhatsApp Bot — *Elimu ya Muungano Ubunifu Challenge*

A WhatsApp bot (Swahili) for a Tanzanian government-run university innovation
contest about the Union of Tanganyika and Zanzibar (*Muungano*). It runs on
FastAPI behind a Twilio WhatsApp webhook, keeps per-user state in SQLite, and
answers free-text questions with Google's Gemini API (free tier) — grounded
strictly in a vetted knowledge base.,

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

Uses `gemini-2.5-flash` (overridable via the `GEMINI_MODEL` env var). Gemini
Flash is fast, capable enough for careful instruction-following Swahili answers
that respect the "only answer from the reference document, stay neutral" rules,
and — critically for this project — available on Google's **free tier**. That
free tier is *rate-limited* (requests per minute / per day), **not**
credit-limited: there is no trial credit to exhaust and no card required. Verify
the current model ID and your live quota in Google AI Studio before deploying,
as free-tier Flash limits have changed more than once.

> **Free-tier data note:** on Google's free Gemini tier, prompts may be used by
> Google to improve their models. That is an accepted trade-off for this
> project's "no paid services" constraint — documented here so it's explicit,
> not hidden.

## Local setup

Requires Python 3.12.

```bash
cd muungano-bot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Get a free Gemini API key from Google AI Studio (aistudio.google.com →
# "Get API key"; no card required). Then provide it (export, or copy the
# example file):
cp .env.example .env      # then edit .env and paste your real key
# export GEMINI_API_KEY=AIza...

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

The suite mocks the Gemini client and isolates the database per test, so it
needs **no** real Twilio or Gemini credentials.

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

## The two WhatsApp channels

The bot serves the **same** conversation logic (`process_message`) over two
independent webhooks — pick whichever channel you connect:

| Channel | Endpoint | Use it for |
| --- | --- | --- |
| **Twilio** | `POST /webhook` | Quick sandbox demos (`join` opt-in, expires 72h). |
| **Meta Cloud API** | `GET`/`POST /meta/webhook` | Free **production** deployment (see below). |

You don't have to choose at the code level — both are always mounted. Only the
Twilio path needs a reply in the HTTP response (TwiML); the Meta path returns
`200` immediately and sends its reply back out via the Graph API.

## Deploying live on WhatsApp for free (Meta Cloud API)

Meta's WhatsApp **Cloud API** is the free production path: Meta hosts the API,
and *service conversations* — where the user messages first and the bot replies
within 24h, which is all this bot ever does — are free and unlimited. (Twilio's
*production* WhatsApp, unlike its sandbox, charges per message, so it is not the
free route.)

1. In **Meta for Developers** → create an app → add the **WhatsApp** product.
   You get a free **test number** immediately (can message up to 5 recipients
   without business verification — enough for a real demo).
2. Copy these into the environment (see `.env.example`):
   - `META_PHONE_NUMBER_ID` — the sender's phone-number ID (not the number).
   - `META_ACCESS_TOKEN` — a token that can call the Graph API to send messages
     (use a long-lived / system-user token for anything beyond testing).
   - `META_VERIFY_TOKEN` — any random string you invent.
3. Deploy the app to a public HTTPS host (see **Hosting** below).
4. In the app's **WhatsApp → Configuration → Webhook**, set the callback URL to
   `https://<your-host>/meta/webhook`, paste the same `META_VERIFY_TOKEN`, and
   **Verify and save** (this triggers the `GET` handshake the bot answers).
   Then **subscribe** the webhook to the `messages` field.
5. Message your number from WhatsApp — you should get the menu back.
   To open it to the general public, complete Meta **Business Verification**
   (free) and register a production number.

> **Number note:** a number currently active in the WhatsApp Business *app*
> can't *simultaneously* run on the Cloud *API* — migrate it (it leaves the
> app) or use a different / the free test number.

## Hosting (always-on, free options)

- **Oracle Cloud "Always Free" VM** — genuinely free forever and always-on (no
  cold starts). Use `deploy/muungano-bot.service` (below). Best fit for a
  permanent deployment.
- **Render free tier** — easiest click-deploy; sleeps after inactivity (~cold
  start on the first message, which Meta/Twilio retry through).
- **Google Cloud Run** — uses the `Dockerfile`, generous free tier, scales to
  zero (needs a card on file even while free).

Both scaffolds below work with either WhatsApp channel:

- **Docker:** `docker build -t muungano-bot .` then
  `docker run -p 8000:8000 -e GEMINI_API_KEY=AIza... muungano-bot`.
  Includes a `HEALTHCHECK` against `/health`. Pass the `META_*` vars with `-e`
  too if serving the Meta channel.
- **VPS + systemd:** see `deploy/muungano-bot.service`. Put secrets in
  `/etc/muungano-bot.env` (`GEMINI_API_KEY=...`, plus the `META_*` vars if used;
  `chmod 600`), install the app under `/opt/muungano-bot` with a virtualenv,
  then `systemctl enable --now muungano-bot`.

## Known limitations

- **Twilio is sandbox-only here.** The Twilio channel targets the WhatsApp
  Sandbox (`join` opt-in, expires after 72h) — fine for demos. For a free
  *production* deployment use the **Meta Cloud API** channel instead (see
  "Deploying live on WhatsApp for free"); reaching the general public still
  requires free Meta Business Verification.
- **Starter content only.** 5 quiz questions and 10 facts, deliberately
  limited to uncontroversial, well-documented historical material. Verify
  every fact against a primary/official source before final submission.
- **Contested political content is deliberately excluded** from the Q&A
  corpus. When asked about disputed aspects of the Union, the bot deflects
  neutrally (points the user to official sources / their teacher) rather than
  taking a side — an intentional scope decision documented in Section 8 of
  `app/knowledge_base.md`.
