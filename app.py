import asyncio
import os
import time

import matplotlib
matplotlib.use("agg")  # must be set before pyplot import; agg is file-only, thread-safe

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import financial_analysis as fa

app = FastAPI(title="SEC Financial Analyzer")

_BASE    = os.path.dirname(os.path.abspath(__file__))
_REPORTS = "/tmp/reports" if os.environ.get("VERCEL") else os.path.join(_BASE, "reports")
_STATIC  = os.path.join(_BASE, "static")

app.mount("/static", StaticFiles(directory=_STATIC), name="static")


class AnalyzeRequest(BaseModel):
    ticker: str


@app.get("/")
def index():
    return FileResponse(os.path.join(_STATIC, "index.html"))


@app.post("/analyze")
async def analyze(body: AnalyzeRequest):
    ticker = body.ticker.strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol or company name is required.")
    try:
        return await asyncio.to_thread(_run_analysis, ticker)
    except fa.TickerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except fa.SECDataError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except fa.InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")


def _run_analysis(ticker: str) -> dict:
    cik, ticker = fa.resolve_company(ticker)
    facts_data = fa.get_company_facts(cik)
    time.sleep(0.5)  # SEC rate limiting

    df = fa.build_ratio_table(facts_data)
    flags = fa.flag_risks(df)
    fa.generate_report(ticker, df, flags, _REPORTS)

    latest = df.iloc[-1]
    kpis = {
        "revenue":       fa.fmt_currency(latest["revenue"]),
        "net_margin":    fa.fmt_pct(latest["net_margin"]),
        "roe":           fa.fmt_pct(latest["roe"]),
        "current_ratio": fa.fmt_ratio(latest["current_ratio"]),
        "latest_year":   int(latest["year"]),
    }

    history = [
        {
            "year":           int(r["year"]),
            "revenue":        fa.fmt_currency(r["revenue"]),
            "net_income":     fa.fmt_currency(r["net_income"]),
            "net_margin":     fa.fmt_pct(r["net_margin"]),
            "current_ratio":  fa.fmt_ratio(r["current_ratio"]),
            "debt_to_equity": fa.fmt_ratio(r["debt_to_equity"]),
            "roe":            fa.fmt_pct(r["roe"]),
            "roa":            fa.fmt_pct(r["roa"]),
        }
        for _, r in df.iterrows()
    ]

    _RED_KEYWORDS = {"Negative net income", "Current ratio below 1.0"}
    formatted_flags = [
        {
            "year":     year,
            "message":  msg,
            "severity": "red" if any(kw in msg for kw in _RED_KEYWORDS) else "amber",
        }
        for year, msg in flags
    ]

    return {"ticker": ticker, "kpis": kpis, "history": history, "flags": formatted_flags}


@app.get("/report/{ticker}")
async def download_report(ticker: str):
    ticker = ticker.upper().strip()
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
