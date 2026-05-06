# ZenZone AI Panel — Deployment Guide

## What you need (takes ~20 minutes total)

1. A free [GitHub](https://github.com) account
2. A free [Render.com](https://render.com) account
3. API keys from each AI service (links below)

---

## Step 1 — Get your API keys

| Service | Where to get it | Cost |
|---------|----------------|------|
| **Claude** (Anthropic) | console.anthropic.com → API Keys | Pay-per-use (tiny) |
| **ChatGPT** (OpenAI) | platform.openai.com → API Keys | Pay-per-use (tiny) |
| **Grok** (xAI) | console.x.ai → API Keys | Pay-per-use (tiny) |

Keep these keys private — treat them like passwords.

---

## Step 2 — Put the code on GitHub

1. Go to github.com and create a new repository called `zenzone-ai-panel`
2. Upload all the files from this folder into that repository
3. Make sure the repository includes: `main.py`, `database.py`, `debate.py`, `file_handler.py`, `requirements.txt`, `render.yaml`, and the `static/` folder

---

## Step 3 — Deploy on Render

1. Go to [render.com](https://render.com) and sign up (free)
2. Click **New → Blueprint**
3. Connect your GitHub account and select your `zenzone-ai-panel` repository
4. Render will auto-detect the `render.yaml` file and set everything up
5. When prompted, add your **Environment Variables**:
   - `ANTHROPIC_API_KEY` = your Claude key
   - `OPENAI_API_KEY` = your ChatGPT key
   - `GROK_API_KEY` = your Grok key
6. Click **Apply** — Render will build and deploy automatically (takes ~3 minutes)
7. Your app URL will be something like `https://zenzone-ai-panel.onrender.com`

---

## Step 4 — Open on phone & desktop

The app works in any browser — just open the URL on your phone or computer.

**Optional:** On iPhone, open the URL in Safari → tap the Share button → "Add to Home Screen" — it'll look and feel like a native app.

---

## Updating the app

Whenever you want to make changes:
1. Edit the files
2. Push to GitHub
3. Render automatically redeploys in ~2 minutes

---

## Costs

Render's free tier keeps your app running with some limitations (it may sleep after 15 minutes of inactivity — first load after sleep takes ~30 seconds). Render's paid tier ($7/month) keeps it always on.

API costs are very small — roughly $0.01–0.05 per question depending on length.

---

## Questions?

All files in this project were built specifically for ZenZone Renovations. Each file has comments explaining what it does.
