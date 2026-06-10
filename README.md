# QuantStream Dashboard

A full-stack AI-powered quantitative finance dashboard that combines live US equities analytics, Hidden Markov Model market regime detection, and a Gemini-grounded strategy chatbot — all deployable for free.

---

## Features

- **Interactive OHLCV charting** — Candlestick charts with SMA-200 overlay and colour-coded HMM regime bands rendered in Plotly
- **Comprehensive technical indicators** — MACD, RSI-14, rolling volatility (5d/21d), Garman-Klass volatility, and 200-day SMA Z-score
- **3-state HMM regime detection** — Gaussian Hidden Markov Model classifies market conditions into Low-Volatility Bull, High-Volatility Bear, and Transitional/Ranging states
- **Risk analytics** — Historical and parametric CVaR at the 95th percentile, plus a full regime statistics heatmap
- **ETF analysis tab** — Curated preset groups across Broad Market, Sectors, Fixed Income, Commodities, and International categories with hover tooltips
- **AI strategy chatbot** — Sidebar assistant powered by Gemini 2.5 Flash, grounded in live dashboard state (regime, RSI, MACD, CVaR) via RAG-augmented context from Qdrant
- **Server-Sent Event streaming** — Token-by-token chat streaming over SSE so responses appear progressively
- **Rate-limit resilience** — Automatic retry with API-reported delay on Gemini 429 errors; transparent to the user
- **Zero-cost deployment** — Designed around Render Free Tier (backend) and Streamlit Community Cloud (frontend)

---

## Prerequisites & Tech Stack

### External accounts required

| Service | Purpose | Free tier |
|---|---|---|
| [Google AI Studio](https://aistudio.google.com/) | Gemini 2.5 Flash (chat + embeddings) | 15 RPM / 1 500 RPD |
| [Qdrant Cloud](https://cloud.qdrant.io/) | Vector store for RAG memory | 1 free cluster, 1 GB RAM |
| [Render](https://render.com/) | FastAPI backend hosting | 750 hrs/month |
| [Streamlit Community Cloud](https://streamlit.io/cloud) | Frontend hosting | Unlimited public apps |
| [GitHub](https://github.com/) | Source repo (both platforms deploy from it) | Free public repos |

### Runtime

- **Python 3.11** (required — both platforms must be configured to use 3.11)

### Backend dependencies  (`requirements.txt` — repo root)

```
fastapi==0.136.3
uvicorn[standard]==0.48.0
python-multipart==0.0.29
pydantic==2.13.4
httpx==0.28.1
yfinance==1.4.1
pandas==3.0.2
numpy==2.4.4
scipy==1.17.1
hmmlearn==0.3.3
scikit-learn==1.8.0
google-genai==2.7.0
qdrant-client==1.18.0
```

### Frontend dependencies (`frontend/requirements.txt`)

```
streamlit==1.58.0
plotly==6.7.0
httpx==0.28.1
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/quantstream-dashboard.git
cd quantstream-dashboard
```

### 2. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install backend dependencies

```bash
pip install -r requirements.txt
```

### 4. Install frontend dependencies

```bash
pip install -r frontend/requirements.txt
```

### 5. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your real credentials:

```bash
# .env  —  never commit this file
GEMINI_API_KEY=AIzaSy_YOUR_KEY_HERE
QDRANT_URL=https://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.us-east4-0.gcp.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here
CORS_ORIGINS=http://localhost:8501
LOG_LEVEL=INFO
```

### 6. Configure the Streamlit frontend secret

```bash
mkdir -p frontend/.streamlit
cat > frontend/.streamlit/secrets.toml << 'EOF'
BACKEND_URL = "http://localhost:8000"
EOF
```

---

## Usage Examples

### Starting the full stack locally

Open two terminals from the repo root:

```bash
# Terminal 1 — FastAPI backend (http://localhost:8000)
set -a && source .env && set +a
uvicorn backend.main:app --reload --port 8000
```

```bash
# Terminal 2 — Streamlit frontend (http://localhost:8501)
cd frontend
streamlit run app.py
```

The Streamlit UI opens automatically in your browser.

---

### Calling the analytics endpoint directly

```bash
# Pull a full analytics payload for NVDA over the past year
curl "http://localhost:8000/api/v1/analytics/NVDA?period=1y" | python -m json.tool
```

Example response shape:

```json
{
  "ticker": "NVDA",
  "period": "1y",
  "ohlcv": [
    { "date": "2024-06-10", "open": 120.4, "high": 135.2, "low": 119.8, "close": 131.5, "volume": 412000000 }
  ],
  "time_series": {
    "dates": ["2024-06-10", "..."],
    "rsi_14": [58.3, "..."],
    "macd": [1.24, "..."],
    "macd_signal": [0.95, "..."],
    "rolling_vol_21d": [0.028, "..."],
    "sma_zscore_200d": [1.42, "..."]
  },
  "regime": {
    "current_regime": "Low-Volatility Bull",
    "regime_state_id": 0,
    "sequence": [0, 0, 1, 2, 0, "..."]
  },
  "risk": {
    "cvar_95_historical": -0.0312,
    "cvar_95_parametric": -0.0298
  }
}
```

---

### Streaming a chat response

```bash
# Stream a strategy question grounded in AAPL's current dashboard state
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Given the current HMM regime and RSI reading, should I be reducing position size?",
    "dashboard_state": {
      "ticker": "AAPL",
      "current_regime": "High-Volatility Bear",
      "regime_state_id": 1,
      "rsi_14": 34.2,
      "macd_histogram": -0.48,
      "rolling_vol_21d": 0.031,
      "gk_vol_21d": 0.029,
      "cvar_95": -0.041,
      "last_close": 178.52,
      "sma_zscore_200d": -1.1,
      "data_as_of": "2025-06-09"
    },
    "session_id": "demo-session-001",
    "history": []
  }'
```

SSE frames arrive as:

```
data: {"type": "token", "content": "Given"}
data: {"type": "token", "content": " the High-Volatility"}
...
data: {"type": "done", "content": ""}
```

---

### Checking backend health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "services": {
    "api": true,
    "gemini_key_set": true,
    "qdrant_url_set": true,
    "qdrant_reachable": true
  }
}
```

---

## Project Structure

```
quantstream-dashboard/
│
├── backend/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app factory, CORS, lifespan
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py            # Analytics + SSE chat endpoints
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── engine.py            # Log returns, MACD, RSI, GK vol, CVaR
│   │   └── regimes.py           # 3-state Gaussian HMM classifier
│   └── services/
│       ├── llm_agent.py         # Gemini 2.5 Flash streaming agent
│       └── vector_db.py         # Qdrant async client for RAG memory
│
├── frontend/
│   ├── app.py                   # Streamlit UI (tabs, charts, sidebar chatbot)
│   ├── api_client.py            # httpx REST + SSE client
│   ├── requirements.txt
│   └── .streamlit/
│       ├── secrets.toml         # BACKEND_URL  (gitignored)
│       └── config.toml          # Theme settings (committed)
│
├── requirements.txt             # Backend deps for Render
├── render.yaml                  # Render declarative service config
├── .env.example                 # Credential template (committed)
├── .gitignore
└── README.md
```

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health + dependency status |
| `GET` | `/api/v1/analytics/{ticker}` | Full analytics payload (OHLCV, indicators, regime, risk) |
| `POST` | `/api/v1/chat/stream` | SSE streaming chat with Gemini |
| `GET` | `/docs` | Interactive Swagger UI (FastAPI auto-generated) |

**Analytics query parameters:**

| Param | Default | Options |
|---|---|---|
| `period` | `1y` | `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y` |

---

## Scripts

| Command | What it does |
|---|---|
| `uvicorn backend.main:app --reload --port 8000` | Start FastAPI backend with hot-reload |
| `cd frontend && streamlit run app.py` | Start Streamlit frontend |
| `curl http://localhost:8000/health` | Verify backend is reachable |
| `curl http://localhost:8000/api/v1/analytics/SPY?period=2y` | Smoke-test the analytics pipeline |
| `python -c "import backend.analytics.engine; print('OK')"` | Verify backend imports resolve |
| `python -c "import streamlit, plotly, httpx; print('OK')"` | Verify frontend imports resolve |

---

## Deployment

### Backend → Render

1. Push to GitHub (public repo works on Render free tier without extra permissions).
2. Create a new **Web Service** in the Render dashboard, connect the repo.
3. Set **Runtime** = Python, **Build Command** = `pip install -r requirements.txt`, **Start Command** = `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
4. Add environment variables in the Render dashboard: `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `CORS_ORIGINS`, `LOG_LEVEL`.
5. Set `PYTHON_VERSION` = `3.11.0`.
6. Render will assign a URL like `https://quantstream-api-xxxx.onrender.com`.

### Frontend → Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
2. Set **Main file path** = `frontend/app.py`, **Python version** = `3.11`.
3. Under **Advanced settings → Secrets**, paste:
   ```toml
   BACKEND_URL = "https://quantstream-api-xxxx.onrender.com"
   ```
4. Deploy. Once the Streamlit URL is assigned, add it (without trailing slash) to `CORS_ORIGINS` in Render and redeploy.

### Cold-start mitigation (Render free tier)

Render free services sleep after 15 minutes of inactivity. Set up a free [UptimeRobot](https://uptimerobot.com) monitor to ping `https://quantstream-api-xxxx.onrender.com/health` every **14 minutes** to keep the service warm.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ | Google AI Studio API key (covers Gemini 2.5 Flash + text-embedding-004) |
| `QDRANT_URL` | ✅ | Full HTTPS URL of your Qdrant Cloud cluster — no trailing slash |
| `QDRANT_API_KEY` | ✅ | Qdrant Cloud API key |
| `CORS_ORIGINS` | ✅ | Comma-separated list of allowed origins (include your Streamlit URL) |
| `LOG_LEVEL` | ❌ | Python log level — defaults to `INFO` |
| `BACKEND_URL` | ✅ (frontend) | Set in Streamlit secrets — URL of the deployed FastAPI service |

---

## Architecture & Data Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT FRONTEND                              │
│  ┌──────────────┐   ┌──────────────────────┐   ┌───────────────────┐  │
│  │  Equity Tab  │   │     ETF Analysis Tab  │   │  Sidebar Chatbot  │  │
│  │  (Plotly)    │   │     (Plotly)          │   │  (st.chat_input)  │  │
│  └──────┬───────┘   └──────────┬───────────┘   └────────┬──────────┘  │
│         │                      │                         │             │
│         └──────────────────────┴──── api_client.py ──────┘             │
│                                      │        │                        │
│                              REST GET │        │ POST + SSE stream      │
└──────────────────────────────────────┼────────┼────────────────────────┘
                                       │        │
                    ┌──────────────────┼────────┼──────────────────────┐
                    │   FASTAPI BACKEND (Render Free Tier)              │
                    │                  │        │                       │
                    │         routes.py│        │routes.py              │
                    │    GET /analytics/{ticker} │POST /chat/stream     │
                    │                  │        │                       │
                    │   ┌──────────────▼──┐  ┌──▼─────────────────┐    │
                    │   │  engine.py      │  │  llm_agent.py       │    │
                    │   │  ─ Log returns  │  │  ─ System prompt    │    │
                    │   │  ─ MACD/RSI     │  │    construction     │    │
                    │   │  ─ GK vol       │  │  ─ RAG retrieval    │    │
                    │   │  ─ CVaR         │  │  ─ Gemini 2.5 Flash │    │
                    │   └──────────────┬──┘  │    streaming        │    │
                    │                  │     └──────┬──────────────┘    │
                    │   ┌──────────────▼──┐         │                   │
                    │   │  regimes.py     │  ┌──────▼──────────────┐    │
                    │   │  ─ 3-state HMM  │  │  vector_db.py       │    │
                    │   │  ─ Semantic      │  │  ─ embed + upsert   │    │
                    │   │    labelling     │  │  ─ cosine search    │    │
                    │   └─────────────────┘  └──────┬──────────────┘    │
                    │                               │                   │
                    └───────────────────────────────┼───────────────────┘
                                                    │
                    ┌───────────────────────────────┼───────────────────┐
                    │    EXTERNAL SERVICES           │                   │
                    │                               │                   │
                    │   ┌───────────────┐   ┌───────▼──────────────┐    │
                    │   │  yfinance     │   │   Qdrant Cloud        │    │
                    │   │  (free OHLCV) │   │   quantstream_memory  │    │
                    │   └───────────────┘   │   768-dim, cosine     │    │
                    │                       └──────────────────────┘    │
                    │   ┌─────────────────────────────────────────┐      │
                    │   │  Google AI Studio (Gemini API)           │      │
                    │   │  ─ gemini-2.5-flash  (chat completion)  │      │
                    │   │  ─ text-embedding-004 (768-dim vectors)  │      │
                    │   └─────────────────────────────────────────┘      │
                    └───────────────────────────────────────────────────┘
```

---

## How It Works

### Quantitative analytics pipeline

When you click **⚡ Analyze**, the frontend sends `GET /api/v1/analytics/{ticker}?period=1y` to the backend. The backend runs this pipeline inside `asyncio.to_thread()` to avoid blocking the async event loop:

1. **yfinance** downloads daily OHLCV data. MultiIndex columns are flattened with `droplevel(level=1, axis=1)` before any column access.
2. **`QuantEngine`** computes log returns, rolling volatility (5d, 21d), Garman-Klass volatility, RSI-14, MACD (12/26/9), 200-day SMA Z-score, and CVaR at 95% (both historical simulation and parametric Gaussian).
3. **`RegimeClassifier`** fits a 3-state Gaussian HMM on a multi-feature observation vector `[log_return, vol_5d, vol_21d]` over the full history, then semantically labels states post-hoc by their mean return and volatility profile.
4. The response is capped at 504 chart points to keep JSON payload sizes manageable over the network.

### AI chatbot and RAG context injection

When a user sends a message in the sidebar chat:

1. The frontend silently captures live dashboard state (ticker, regime, RSI, MACD, CVaR, last close, data date) and POSTs it to `POST /api/v1/chat/stream` alongside the user message and conversation history.
2. The backend embeds the user message with `text-embedding-004` (768 dimensions) and runs a cosine similarity search in the Qdrant `quantstream_memory` collection to retrieve the top relevant prior conversation snippets (RAG context).
3. **`GeminiAgent`** assembles a structured system prompt that includes the live metrics table, semantic regime interpretation, RAG snippets, and behavioural guidelines for grounded financial commentary.
4. The response is streamed token-by-token from Gemini 2.5 Flash back to the backend, which forwards each token as an SSE frame (`data: {"type": "token", "content": "..."}`).
5. After the stream closes, the full exchange is embedded and upserted into Qdrant for future retrieval.

### Memory management (512 MB Render constraint)

All heavy libraries — `pandas`, `numpy`, `hmmlearn`, `scikit-learn`, `google-genai`, `qdrant-client` — are imported **inside** the endpoint functions, never at module scope. FastAPI idles at ~80 MB RAM and peaks at ~350 MB during a full analytics + chat cycle, staying within Render's 512 MB free-tier ceiling.

---

## Known Limitations

**Render free tier sleeps after 15 minutes of inactivity.** The first request after a sleep incurs a ~30 second cold start. The health endpoint is designed to be cheap (no analytics libraries loaded) so UptimeRobot pings incur minimal overhead.

**yfinance data is delayed ~15 minutes for most tickers.** This is sufficient for end-of-day analysis and swing trading contexts but unsuitable for intraday strategies.

**HMM state labelling is unsupervised.** The semantic labels ("Low-Volatility Bull", "High-Volatility Bear", "Transitional/Ranging") are assigned post-hoc using heuristic thresholds on each state's mean return and volatility. The HMM does not know what a "bull market" is — it finds statistical clusters in return/volatility space. Always verify the regime interpretation against the candlestick chart.

**Gemini free tier rate limits.** Google AI Studio free tier permits 15 requests per minute for Gemini 2.5 Flash. Each chat message costs one request. The backend implements an automatic retry with the API-reported delay on `429 RESOURCE_EXHAUSTED` errors; responses are transparent to the user but may take 25–35 seconds to begin streaming if the limit has been hit.

**Qdrant free cluster.** The single free Qdrant cluster has 1 GB RAM. The `quantstream_memory` collection stores 768-dimensional float32 vectors, so you can accumulate approximately 300,000 chat turns before needing to clean up or upgrade.

**No authentication.** The API endpoints have no authentication layer. Do not add sensitive personal financial data to the chatbot history if the deployment is publicly accessible.

---

## Troubleshooting

### Backend shows `{"status": "ok"}` but charts don't load

Check that `CORS_ORIGINS` in Render includes your Streamlit URL **without a trailing slash**. A trailing slash is the most common cause of CORS rejections on this stack.

```bash
# Correct
CORS_ORIGINS=http://localhost:8501,https://your-app.streamlit.app

# Incorrect — note the trailing slash
CORS_ORIGINS=http://localhost:8501,https://your-app.streamlit.app/
```

### `ModuleNotFoundError: No module named 'backend'`

The FastAPI server must be launched from the **repo root**, not from inside the `backend/` directory. The start command `uvicorn backend.main:app` resolves `backend` as a package relative to the working directory.

```bash
# Correct — from repo root
uvicorn backend.main:app --reload --port 8000

# Incorrect
cd backend && uvicorn main:app --reload  # breaks the import chain
```

### Chat returns `404 NOT_FOUND: models/gemini-X is not found`

The `gemini-2.5-flash` model is only available to Google AI Studio API keys created after its launch. Keys tied to billing-enabled Google Cloud projects may have free-tier quotas set to zero. Create a fresh API key at [aistudio.google.com](https://aistudio.google.com) (not the Google Cloud Console) and replace `GEMINI_API_KEY` in Render.

### HMM fit raises `ValueError: rows of transmat_ must sum to 1`

This occurs when the price series is too short for the HMM to converge. The minimum required is ~60 trading days. Switch the period selector to `6mo` or longer. The `1mo` period (~21 trading days) is not sufficient for stable 3-state fitting.

### Streamlit shows "Backend: timeout" after the backend is confirmed running

The health check is cached when `"ok"`. A `"timeout"` on first load (Render cold start) gets cached and won't re-probe automatically unless you close and reopen the browser tab. The re-probe logic triggers on every rerun when the last known status is not `"ok"`, so clicking anything interactive in the sidebar forces a re-check within a second.

### `qdrant_reachable: false` in the health response

Verify that `QDRANT_URL` has no trailing slash and starts with `https://`. The Qdrant Cloud cluster URL format is:

```
https://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.us-east4-0.gcp.cloud.qdrant.io
```

Also confirm the `QDRANT_API_KEY` matches the key shown in the Qdrant Cloud dashboard under the **API Keys** tab for your cluster.

---

## Contributing

Contributions are welcome. Please open an issue first to discuss the change before submitting a pull request.

Before submitting:

1. Ensure all new backend code keeps heavy imports **inside** the function bodies — never at module scope — to respect the 512 MB RAM constraint.
2. All new API endpoints should include Pydantic input/output models with full type hints.
3. Frontend changes must not import any heavy analytics libraries into `app.py` or `api_client.py` — the Streamlit process should stay lightweight.
