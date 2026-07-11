import asyncio
import os
import re
import time

import matplotlib
matplotlib.use("agg")  # must be set before pyplot import; agg is file-only, thread-safe

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from collections import defaultdict
import requests

import financial_analysis as fa

# ── LOAD ENV FILE MANUALLY ──
def load_dotenv():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        parts = line.split("=", 1)
                        key = parts[0].strip()
                        val = parts[1].strip()
                        if val.startswith(('"', "'")) and val.endswith(('"', "'")):
                            val = val[1:-1]
                        os.environ[key] = val

load_dotenv()

app = FastAPI(title="SEC Financial Analyzer")

_BASE    = os.path.dirname(os.path.abspath(__file__))
_REPORTS = "/tmp/reports" if os.environ.get("VERCEL") else os.path.join(_BASE, "reports")
_STATIC  = os.path.join(_BASE, "static")

app.mount("/static", StaticFiles(directory=_STATIC), name="static")


class AnalyzeRequest(BaseModel):
    ticker: str
    period: str = "annual"  # "annual" or "quarterly"


@app.get("/")
def index():
    return FileResponse(os.path.join(_STATIC, "index.html"))


@app.post("/analyze")
async def analyze(body: AnalyzeRequest):
    ticker = body.ticker.strip()
    period = body.period.strip().lower()
    if period not in ("annual", "quarterly"):
        period = "annual"
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol or company name is required.")
    try:
        return await asyncio.to_thread(_run_analysis, ticker, period)
    except fa.TickerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except fa.SECDataError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except fa.InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")


def _run_analysis(ticker: str, period: str = "annual") -> dict:
    cik, ticker = fa.resolve_company(ticker)
    facts_data = fa.get_company_facts(cik)
    time.sleep(0.5)  # SEC rate limiting

    df = fa.build_ratio_table(facts_data, period_type=period)
    flags = fa.flag_risks(df)
    fa.generate_report(ticker, df, flags, _REPORTS)

    latest = df.iloc[-1]
    latest_year = str(latest["year"])
    if latest_year.endswith(".0"):
        latest_year = latest_year[:-2]

    kpis = {
        "revenue":       fa.fmt_currency(latest["revenue"]),
        "net_margin":    fa.fmt_pct(latest["net_margin"]),
        "roe":           fa.fmt_pct(latest["roe"]),
        "current_ratio": fa.fmt_ratio(latest["current_ratio"]),
        "latest_year":   latest_year,
    }

    history = []
    for _, r in df.iterrows():
        year_str = str(r["year"])
        if year_str.endswith(".0"):
            year_str = year_str[:-2]
        history.append({
            "year":           year_str,
            "revenue":        fa.fmt_currency(r["revenue"]),
            "net_income":     fa.fmt_currency(r["net_income"]),
            "net_margin":     fa.fmt_pct(r["net_margin"]),
            "current_ratio":  fa.fmt_ratio(r["current_ratio"]),
            "debt_to_equity": fa.fmt_ratio(r["debt_to_equity"]),
            "roe":            fa.fmt_pct(r["roe"]),
            "roa":            fa.fmt_pct(r["roa"]),
        })

    _RED_KEYWORDS = {"Negative net income", "Current ratio below 1.0"}
    formatted_flags = []
    for year, msg in flags:
        year_str = str(year)
        if year_str.endswith(".0"):
            year_str = year_str[:-2]
        formatted_flags.append({
            "year":     year_str,
            "message":  msg,
            "severity": "red" if any(kw in msg for kw in _RED_KEYWORDS) else "amber",
        })

    return {"ticker": ticker, "kpis": kpis, "history": history, "flags": formatted_flags}


_TICKER_RE = re.compile(r'^[A-Z0-9.\-]{1,10}$')

@app.get("/report/{ticker}")
async def download_report(ticker: str):
    ticker = ticker.upper().strip()
    if not _TICKER_RE.match(ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker symbol.")
    pdf_path = os.path.join(_REPORTS, f"{ticker}_report.pdf")
    if not os.path.exists(pdf_path):
        # /tmp is ephemeral on serverless platforms — regenerate if not cached
        try:
            await asyncio.to_thread(_run_analysis, ticker)
        except fa.TickerNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except fa.SECDataError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        except fa.InsufficientDataError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{ticker}_financial_report.pdf",
    )


# ── AI AGENT (WARREN BUFFETT PERSONA) INTEGRATION ──

class ChatRequest(BaseModel):
    message: str
    company_context: str | None = None  # validated at endpoint level


# Sliding-window rate limiter records: client_ip -> list of timestamps
_rate_limit_records = defaultdict(list)

def check_rate_limit(client_ip: str, max_requests: int = 6, window_seconds: int = 60):
    now = time.time()
    # Keep only timestamps in the current window
    _rate_limit_records[client_ip] = [t for t in _rate_limit_records[client_ip] if now - t < window_seconds]
    if len(_rate_limit_records[client_ip]) >= max_requests:
        raise HTTPException(
            status_code=429,
            detail="Whoa there, partner! You're asking questions faster than I can read financial statements. "
                   "Take a breath and try again in a minute."
        )
    _rate_limit_records[client_ip].append(now)


def _generate_buffett_response(prompt: str, context_summary: str = "", api_key: str = None) -> str:
    if not api_key:
        return (
            "Well, my assistant Becky tells me we don't have the GEMINI_API_KEY set up in our computer system "
            "here in Omaha. But if I had to give you some advice: look for businesses with enduring competitive moats, "
            "run by honest and capable managers, and selling at a sensible price. Without that key, I can't analyze "
            "this specific business for you, but the basic principles of value investing never change."
        )
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    
    system_instruction = (
        "You are Warren Buffett, the legendary value investor and Chairman of Berkshire Hathaway. "
        "Answer financial, business, and investing questions using your famous folksy wisdom. Speak in the first person ('I', 'we' at Berkshire). "
        "Use the real-time Google Search results to incorporate latest stock context, market trends, or big news if available. "
        "\n\nIMPORTANT RULES FOR RESPONSE LENGTH & FORMAT:"
        "\n1. By default, keep your response extremely short, simple, and direct—exactly 2 to 3 sentences max! Do NOT use any bullet points, lists, or headers in this default summary mode."
        "\n2. If the user explicitly asks for 'more details', 'breakdown', or 'numbers' (like clicking the more details button), then write a structured breakdown using bullet points:"
        "\n* **Moat & News Trends**: (Briefly evaluate margins and latest news/trends)"
        "\n* **Profitability & ROE**: (Briefly evaluate capital returns)"
        "\n* **Debt & Safety**: (Briefly evaluate leverage and liquidity)"
        "\n* **Warren's Verdict**: (Explain folksily whether you would purchase it and why)."
        "\n\nStrictly refuse to answer questions unrelated to business, finance, career advice, or economics. "
        "If asked about coding or non-financial topics, politely refuse in your Buffett persona. "
        "Never give direct buy/sell advice; always state that it is for educational purposes."
    )
    
    contents = []
    if context_summary:
        contents.append({
            "role": "user",
            "parts": [{"text": f"Here is the financial data for the company we are looking at:\n{context_summary}"}]
        })
        contents.append({
            "role": "model",
            "parts": [{"text": "I have reviewed those financial figures. What is your question about this business?"}]
        })
        
    contents.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })
    
    body = {
        "contents": contents,
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "tools": [
            {"google_search": {}}
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048,
            "thinkingConfig": {
                "thinkingBudget": 0
            }
        }
    }
    
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        # Fallback: if search tools failed (e.g. 429 rate limit on search grounding for free tier keys), retry without tools
        if "tools" in body:
            del body["tools"]
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=30)
                resp.raise_for_status()
                result = resp.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
            except requests.exceptions.HTTPError as he2:
                if he2.response is not None and he2.response.status_code == 429:
                    return (
                        "Excuse me, it seems our telegraph wires to the registry are temporarily jammed with too many requests "
                        "(Google API Error 429). Let's wait a minute for the lines to clear and ask me again!"
                    )
                return f"Excuse me, I'm having some trouble connecting to the wires: {he2}. Let's try again in a bit."
            except Exception as retry_err:
                return f"Excuse me, I'm having some trouble connecting to the wires: {retry_err}. Let's try again in a bit."
        
        # If it failed and we had no tools, check if it was a 429
        if isinstance(e, requests.exceptions.HTTPError) and e.response is not None and e.response.status_code == 429:
            return (
                "Excuse me, it seems our telegraph wires to the registry are temporarily jammed with too many requests "
                "(Google API Error 429). Let's wait a minute for the lines to clear and ask me again!"
            )
        return f"Excuse me, I'm having some trouble connecting to the wires: {e}. Let's try again in a bit."


@app.post("/chat")
async def chat(body: ChatRequest, fastapi_req: Request):
    client_ip = fastapi_req.client.host if fastapi_req.client else "unknown"
    # Rate limit check: 6 queries per minute
    check_rate_limit(client_ip, max_requests=6, window_seconds=60)
    
    # Input length and abuse checks
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(message) > 400:
        raise HTTPException(status_code=400, detail="Message is too long. Limit is 400 characters.")
    if body.company_context and len(body.company_context) > 5000:
        raise HTTPException(status_code=400, detail="Company context is too long.")
        
    # Prevent injection attacks/abuse
    abuse_keywords = ["ignore previous", "system prompt", "translate this", "write a python", "jailbreak"]
    for kw in abuse_keywords:
        if kw in message.lower():
            raise HTTPException(
                status_code=400,
                detail="Folks, let's keep our discussion on investing and businesses, just like we do at Berkshire."
            )
            
    api_key = os.environ.get("GEMINI_API_KEY")
    response_text = await asyncio.to_thread(
        _generate_buffett_response,
        message,
        body.company_context or "",
        api_key
    )
    return {"response": response_text}
