# Agentic Finance

An AI system that looks at a stock and tells you **what to do, how much to invest,
where to place a stop-loss, and why** — with the risk limits enforced in code rather
than suggested by a language model.

Four specialist agents (price trend, company financials, news mood, news meaning) each
give an opinion. Those are combined into one view, and a risk engine turns that view
into a properly sized order. Every decision comes with a plain-English explanation.

> Research and educational project. Simulation only — no real money, no live trading.

---

## Contents

1. [Before you start](#1-before-you-start)
2. [Get the code](#2-get-the-code)
3. [Set up Python](#3-set-up-python)
4. [Install the project](#4-install-the-project)
5. [Check that it works](#5-check-that-it-works)
6. [Run the app](#6-run-the-app)
7. [Optional: add the AI model (Ollama)](#7-optional-add-the-ai-model-ollama)
8. [Optional: API keys](#8-optional-api-keys)
9. [Using the app](#9-using-the-app)
10. [Running the experiments](#10-running-the-experiments)
11. [Troubleshooting](#11-troubleshooting)
12. [What's in each folder](#12-whats-in-each-folder)

---

## 1. Before you start

You need two things installed. Check whether you already have them by opening a
terminal and running:

```bash
python --version
git --version
```

**If `python` is not found or the version is below 3.10**, download it from
<https://www.python.org/downloads/>. On Windows, **tick "Add Python to PATH"** on the
first screen of the installer — this is easy to miss and causes most setup problems.

**If `git` is not found**, download it from <https://git-scm.com/downloads>.

On Windows, use **Git Bash** (installed with Git) for every command in this guide.
On macOS or Linux, use the normal Terminal.

**Good news:** you do **not** need a GPU, an API key, or an AI model to run this. Those
are optional extras covered later. Steps 1–6 give you a fully working system.

---

## 2. Get the code

Pick a folder where you want the project to live, then:

```bash
git clone https://github.com/Thejeswee-N-20/Agentic_finance_.git
cd Agentic_finance_
```

Every command from here on assumes you are **inside this folder**.

---

## 3. Set up Python

We create a "virtual environment" — a private folder for this project's libraries, so
nothing on your computer is affected.

**Create it (all systems):**

```bash
python -m venv .venv
```

**Activate it — pick the line for your system:**

```bash
source .venv/Scripts/activate      # Windows (Git Bash)
```
```bash
.venv\Scripts\activate             # Windows (Command Prompt / PowerShell)
```
```bash
source .venv/bin/activate          # macOS / Linux
```

You will know it worked because your prompt now starts with `(.venv)`.

> **Remember:** every time you open a new terminal to use this project, you must
> activate the environment again with the same command.

---

## 4. Install the project

```bash
pip install -e ".[dev]"
```

This downloads everything the project needs. It takes **2–5 minutes** and prints a lot
of text — that is normal. You are looking for `Successfully installed ...` at the end.

---

## 5. Check that it works

```bash
python -m pytest -q
```

Expected output (takes about 30 seconds):

```
326 passed
```

If you see that, the installation is correct. These tests run entirely offline — no
internet, no API keys — so they are a reliable check.

---

## 6. Run the app

```bash
python run_ui.py
```

Wait about 10 seconds, then open **<http://localhost:8501>** in your browser.

To stop the app, press **Ctrl + C** in the terminal.

### What works right now, with nothing else installed

- Live stock prices and company financials (from Yahoo Finance — no key needed)
- Two of the four agents: **price trend** and **company financials**
- The complete risk engine — position sizing, stop-loss, risk limits
- All backtesting (the **Track Record** tab)
- The full explanation system (the **Why** tab)

The two news-based agents stay inactive until you add an AI model in the next step.
The system does not break — it simply makes decisions using the two agents it has.

---

## 7. Optional: add the AI model (Ollama)

This activates the remaining two agents. Everything runs on your own machine — no
cloud service, no cost, no data leaving your computer.

### Step 7.1 — Install Ollama

Ollama is a free program that runs AI models locally.

1. Go to <https://ollama.com/download>
2. Download the installer for your system and run it
3. **Restart your terminal afterwards**

Check it installed:

```bash
ollama --version
```

If that prints a version number, you are ready. Ollama runs quietly in the background
from now on.

### Step 7.2 — Add a model

You have two options. **Option A** uses the model that was fine-tuned for this project.
**Option B** is a one-command alternative if you do not have that file.

---

**Option A — the project's own fine-tuned model (recommended)**

The model file is about **1.9 GB**, which is too large for GitHub, so it is shared
separately (Google Drive link provided with this project).

1. Download the file `qwen2.5-3b-fingpt-q4_k_m.gguf`

2. Move it into the `fine-tuned-slm` folder inside this project. When you are done,
   that folder must contain both of these side by side:

   ```
   fine-tuned-slm/
     ├── Modelfile                          <- already in the repo
     └── qwen2.5-3b-fingpt-q4_k_m.gguf      <- the file you downloaded
   ```

3. Register it with Ollama:

   ```bash
   cd fine-tuned-slm
   ollama create qwen2.5-3b-fingpt -f Modelfile
   cd ..
   ```

   This takes a minute or two and ends with `success`.

4. Confirm it is there:

   ```bash
   ollama list
   ```

   You should see `qwen2.5-3b-fingpt` in the list.

---

**Option B — a standard model instead**

```bash
ollama pull qwen2.5:3b
```

Downloads about 2 GB. Use this if you do not have the fine-tuned file.

---

### Step 7.3 — Tell the project which model to use

Create your settings file by copying the example:

```bash
cp .env.example .env
```

Open `.env` in any text editor (Notepad is fine) and set these two lines:

```bash
AGENTIC_SLM_PROVIDER=ollama
AGENTIC_OLLAMA_MODEL=qwen2.5-3b-fingpt
```

If you chose **Option B**, use `AGENTIC_OLLAMA_MODEL=qwen2.5:3b` instead.

Save the file, then restart the app (`Ctrl + C`, then `python run_ui.py` again).

### Step 7.4 — Warm the model up first

The **first** AI request loads the model into memory and takes about **40 seconds**.
Every request after that takes 3–6 seconds. To get the slow one out of the way:

```bash
ollama run qwen2.5-3b-fingpt "reply OK"
```

Wait for a reply, then type `/bye` to exit. The model now stays loaded.

---

## 8. Optional: API keys

Only needed for two extra features. The app works fine without them.

Open your `.env` file and add:

```bash
# Historical news search — free developer account at tavily.com
TAVILY_NEWS_API_KEY=tvly-...

# Only if you want to compare against Google's cloud AI
GOOGLE_API_KEY=...
```

Without a Tavily key the app falls back to free Yahoo headlines. Those work for
today's news but cannot be filtered by date, so they are unsuitable for historical
backtests.

Company filings come from the public SEC EDGAR database and need **no key**.

---

## 9. Using the app

Open <http://localhost:8501>. There are five tabs.

**Recommendation** — the main screen. Type a company name in the left sidebar (for
example `Apple`, `Reliance`, `Infosys`), pick it from the dropdown, then click
**Analyze this stock & recommend**. It takes 20–35 seconds and returns a Buy / Sell /
Hold decision, how much of your money to use, a stop-loss price, and the four agents'
individual opinions.

**Track Record** — replays history to show how the strategy would have performed
against simple benchmarks. Needs no AI model.

**Why** — trains an explanation model and shows which factors drove the decisions,
plus what would need to change to flip the answer. Needs no AI model.

**Ideas for You** — finds similar companies and ranks them against your chosen risk
level.

**Help** — a plain-English description of the system.

### Two useful things to try

**Change the risk setting.** In the sidebar, switch between Conservative, Moderate and
Aggressive, then re-run the same stock. The recommended amount changes, because your
risk preference feeds directly into the sizing maths.

**Try an Indian stock.** Search `Reliance` or `Infosys`. Prices appear in rupees. Note
that Indian companies have no SEC filings, so that panel will be empty — the system
handles this and carries on normally.

---

## 10. Running the experiments

These reproduce the project's results from the terminal. The first three need no
internet connection and no API keys.

```bash
python run_backtest_demo.py       # one stock vs simple strategies
python run_regime_eval_demo.py    # five market periods (the headline result)
python run_xai_demo.py            # explanation quality and feature importance
python run_forward_paper_demo.py  # forward test with statistical corrections
```

Try a different stock by putting the symbol in front of the command:

```bash
REPRO_TICKER=RELIANCE.NS python run_backtest_demo.py
```

These need an AI model and API keys:

```bash
python run_gemini_e2e_demo.py     # one full live decision, printed step by step
python run_news_backtest_demo.py  # backtest driven by historical news
python run_rag_ingest_demo.py     # download SEC filings into a local index
```

Check your local setup at any time:

```bash
python verify_local_models.py
```

---

## 11. Troubleshooting

**`python: command not found`**
Python is not installed, or was not added to PATH during installation. Reinstall from
python.org and tick "Add Python to PATH".

**`pip: command not found`, or packages install to the wrong place**
The virtual environment is not active. Run the activate command from Step 3 again. Your
prompt should show `(.venv)`.

**Every stock shows "No price data" (Windows)**
Usually antivirus or a company network inspecting secure connections. The project
handles this automatically, but you can force a refresh:

```bash
rm -rf agentic_finance/_cacerts
```

Then restart the app.

**"Port 8501 is already in use"**
Another copy is still running. Either close it, or use a different port:

```bash
streamlit run ui/app.py --server.port 8502
```

**A recommendation takes 40+ seconds**
The AI model was cold. See Step 7.4 to warm it up. Later requests are much faster.

**`ollama: command not found`**
Ollama is installed but your terminal has not picked it up. Close and reopen the
terminal.

**A company search finds nothing**
The company may have been renamed. For example, Zomato is now listed as `ETERNAL.NS`.
Try the current name, or type the exchange symbol directly. Indian stocks end in `.NS`
(NSE) or `.BO` (BSE).

**The tests fail on a fresh install**
Make sure Step 4 finished with `Successfully installed`. Then run
`pip install -e ".[dev]"` again.

---

## 12. What's in each folder

```
agentic_finance/          the system itself
  agents_v2/              the four agents and the logic that combines them
  risk_engine/            position sizing, stop-loss, VaR/CVaR, portfolio and
                          compliance limits
  rag/                    downloads and searches SEC company filings
  xai/                    the explanation layer (SHAP, counterfactuals)
  backtest/               historical testing and performance measurement
  slm/                    connects to the AI model (local or cloud)
  news/                   fetches news headlines
  decision.py             the whole pipeline in one function — start reading here
  features.py             turns prices into the numbers the agents use

ui/app.py                 the web dashboard
tests/                    326 automated tests
run_*.py                  the experiments and demos
fine-tuned-slm/           scripts used to train the AI model, and the Modelfile
                          needed to load it into Ollama
```

**If you want to read the code**, start with `agentic_finance/decision.py` — it shows
the entire flow from data to final decision in about 100 lines. Then look at
`agentic_finance/risk_engine/risk_manager.py`, which contains the risk rules.

---

## Rebuilding the AI model from scratch

Not required, but included for completeness. Needs a GPU with roughly 24 GB of memory
and takes about 85 minutes.

```bash
cd fine-tuned-slm
pip install torch transformers peft bitsandbytes trl datasets accelerate
python train.py
python merge_adapter.py
```

The result is then converted to GGUF format and loaded into Ollama as in Step 7.2.
