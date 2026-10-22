"""
Analyze Router — /api/analyze
------------------------------
POST /api/analyze   → full phishing analysis of a URL
GET  /api/stats     → system statistics
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional

from ..models.predictor import analyze_url
from ..database.db import log_analysis, get_stats

router = APIRouter(prefix="/api", tags=["analyze"])


class AnalyzeRequest(BaseModel):
    url: str
    fetch_content: Optional[bool] = True


class AnalyzeResponse(BaseModel):
    url: str
    phishing_probability: float
    typosquat_score: float
    risk_level: str
    risk_color: str
    explanations: list
    feature_importances: list
    url_features: dict
    content_features: dict
    model_type: str


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest, request: Request):
    """
    Analyze a URL for phishing risk.
    Returns probability, risk level, SHAP explanations, and feature details.
    """
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    # Add schema if missing
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        result = analyze_url(url, fetch_content=req.fetch_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    # Log to DB
    try:
        log_analysis(
            url=url,
            risk_level=result["risk_level"],
            phishing_prob=result["phishing_probability"],
            typosquat_score=result["typosquat_score"],
            model_type=result["model_type"],
        )
    except Exception:
        pass  # Don't fail the API due to logging errors

    return result


@router.get("/stats")
async def stats():
    """Return system-wide statistics."""
    return get_stats()
