#!/usr/bin/env python3
"""
Utility script to pull factual data for JM Aggressive Hybrid Fund Direct Growth
directly from Groww's public pages.

The script fetches Next.js data from the scheme page, validates the key
attributes the FAQ assistant must answer (expense ratio, exit load, minimum
investment, riskometer, benchmark, tax notes, SID link), and stores the
results as structured JSON alongside the source URL for citation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup

SCHEME_URL = "https://groww.in/mutual-funds/jm-aggressive-hybrid-fund-direct-growth"

ROOT = Path(__file__).resolve().parents[1]
SCHEME_OUTPUT = ROOT / "data" / "schemes" / "jm-aggressive-hybrid-fund-direct-growth.json"


class ScrapeError(RuntimeError):
    """Raised when expected content is missing from the source page."""


def fetch_html(url: str) -> BeautifulSoup:
    """Download a page and return a BeautifulSoup DOM tree."""
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return BeautifulSoup(resp.content, "html.parser")


def load_scheme_blob(soup: BeautifulSoup) -> Dict[str, Any]:
    script = soup.find("script", id="__NEXT_DATA__")
    if script is None or not script.string:
        raise ScrapeError("Unable to locate Next.js data blob on scheme page")
    payload = json.loads(script.string)
    try:
        return payload["props"]["pageProps"]["mf"]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ScrapeError("Unexpected Next.js payload format") from exc


def normalize_money(value: int | float | None) -> Dict[str, Any]:
    if value is None:
        return {"value": None, "display": None}
    return {
        "value": int(value),
        "display": f"₹{int(value):,}",
    }


def extract_risk_label(mf_data: Dict[str, Any]) -> str:
    meta_desc = mf_data.get("meta_desc") or ""
    match = re.search(r"Risk is ([A-Za-z ]+)", meta_desc)
    if match:
        return match.group(1).strip()
    nfo_risk = mf_data.get("nfo_risk")
    if isinstance(nfo_risk, str) and nfo_risk.strip():
        return nfo_risk.strip()
    raise ScrapeError("Riskometer label missing in scheme data")


def format_percentage(raw: str | float | None) -> Dict[str, Any]:
    if raw in (None, ""):
        return {"value": None, "display": None}
    value = float(raw)
    return {"value": value, "display": f"{value:.2f}%"}


def extract_lock_in(mf_data: Dict[str, Any]) -> Dict[str, Any]:
    lock = mf_data.get("lock_in") or {}
    days = lock.get("days")
    months = lock.get("months")
    years = lock.get("years")
    if not any([days, months, years]):
        return {
            "applicable": False,
            "notes": "No lock-in mentioned; scheme is not an ELSS product.",
        }
    total_days = (
        (days or 0)
        + (months or 0) * 30
        + (years or 0) * 365
    )
    return {"applicable": True, "lock_in_days": total_days}


def build_scheme_payload(url: str) -> Dict[str, Any]:
    soup = fetch_html(url)
    mf_data = load_scheme_blob(soup)
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "scheme_key": mf_data.get("search_id") or mf_data.get("scheme_code"),
        "scheme_name": mf_data["scheme_name"],
        "source_url": url,
        "fetched_at": timestamp,
        "metadata": {
            "plan_type": mf_data.get("plan_type"),
            "scheme_type": mf_data.get("scheme_type"),
            "category": mf_data.get("category"),
            "sub_category": mf_data.get("sub_category"),
            "fund_house": (mf_data.get("amc_info") or {}).get("name"),
        },
        "attributes": {
            "minimum_lumpsum_investment": {
                **normalize_money(mf_data.get("min_investment_amount")),
                "source_url": url,
            },
            "minimum_sip_investment": {
                **normalize_money(mf_data.get("min_sip_investment")),
                "source_url": url,
            },
            "expense_ratio": {
                **format_percentage(mf_data.get("expense_ratio")),
                "source_url": url,
            },
            "exit_load": {
                "value": mf_data.get("exit_load"),
                "source_url": url,
            },
            "riskometer": {
                "label": extract_risk_label(mf_data),
                "source_url": url,
            },
            "benchmark": {
                "name": mf_data.get("benchmark_name"),
                "source_url": url,
            },
            "taxation": {
                "summary": (mf_data.get("category_info") or {}).get("tax_impact"),
                "source_url": url,
            },
            "lock_in": {
                **extract_lock_in(mf_data),
                "source_url": url,
            },
        },
        "documents": [],
    }
    sid_url = mf_data.get("sid_url")
    if sid_url:
        payload["documents"].append(
            {
                "type": "SID",
                "url": sid_url,
                "source_url": url,
            }
        )
    description = mf_data.get("description")
    if description:
        payload["metadata"]["objective"] = description.strip()
    return payload


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    scheme_payload = build_scheme_payload(SCHEME_URL)
    write_json(SCHEME_OUTPUT, scheme_payload)
    print(
        "Saved scheme data to",
        SCHEME_OUTPUT.relative_to(ROOT),
    )


if __name__ == "__main__":
    main()
