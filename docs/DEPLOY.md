# Deploying the remote study

The study needs HTTPS (browsers only allow webcam access over https), ~3 GB RAM
(RoBERTa + ResNet-50 on CPU), and the Groq key. One small VM is enough for a
few concurrent participants.

## 1. Server (any Linux VM with Docker, e.g. a university VM, Hetzner, Fly.io machine)

```bash
git clone https://github.com/sainadth/BimodalAffectiveAlignment.git && cd BimodalAffectiveAlignment
cp .env.example .env            # fill GROQ_API_KEY, LLM_PROVIDER=groq, HF_TOKEN, STUDY_DOMAIN=study.example.edu
docker compose up -d --build    # first build ~10 min (torch + model download on first request)
docker compose logs -f study    # wait for "Application startup complete"
```

Point the DNS A-record of `STUDY_DOMAIN` at the VM first; Caddy fetches the
certificate automatically. Check `https://STUDY_DOMAIN/api/health` returns
`"llm_configured": true`.

## 2. Participant links

```bash
docker compose exec study python scripts/make_participant_links.py --n 28 --base https://$STUDY_DOMAIN
```

Prints `p01 https://.../p/<token>` lines. Hand each participant exactly one
link (email or paper). `data/participants.json` is the only mapping from token to
number; keep it with the consent records, never commit it.

## 3. During the study

- Data land in `data/study_responses.csv` and `data/study_pets_global.csv` on
  the host (bind-mounted). Back them up daily; they are the only copy.
- A participant who closes the tab can reopen their link and resumes at the
  first unlogged turn.
- Withdrawal from the debrief page deletes their rows immediately.
- Photos are decoded in RAM and dropped after inference; nothing is written to
  disk. Uvicorn access logs contain the token path — rotate/delete logs after
  the study, or disable access logging with `--no-access-log`.

## 4. Analysis

```bash
docker compose exec study python scripts/analyze_study.py
```

## Local development

```bash
uvicorn server.app:app --reload --port 8000        # backend (loads models on start)
cd web && npm install && npm run dev               # frontend on :5173, proxies /api to :8000
python scripts/make_participant_links.py --n 2      # then open http://localhost:5173/p/<token>
```

Camera works on `localhost` without HTTPS. `npm run build` regenerates
`web/dist`, which the FastAPI app serves when present.
