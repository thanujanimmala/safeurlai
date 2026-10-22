"""
Reports Router — /api/report
-----------------------------
POST /api/report    → submit a user report for a URL
GET  /api/reports   → list recent reports (admin)
POST /api/retrain   → trigger manual retraining
"""

import os
import subprocess
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional

from ..database.db import save_report, get_pending_reports, get_stats

router = APIRouter(prefix="/api", tags=["reports"])

RETRAIN_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "ml", "retrain.py"
)


class ReportRequest(BaseModel):
    url: str
    model_prediction: str           # "SAFE" | "SUSPICIOUS" | "PHISHING"
    user_label: str                 # "PHISHING" | "SAFE"
    phishing_prob: Optional[float] = 0.0
    typosquat_score: Optional[float] = 0.0


@router.post("/report")
async def submit_report(req: ReportRequest, request: Request):
    """
    Advancement #4: Accept user-submitted URL report.
    Stores it in DB for the active learning retraining pipeline.
    """
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    reporter_ip = request.client.host if request.client else ""

    try:
        save_report(
            url=url,
            model_prediction=req.model_prediction,
            user_label=req.user_label,
            phishing_prob=req.phishing_prob,
            typosquat_score=req.typosquat_score,
            reporter_ip=reporter_ip,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save report: {str(e)}")

    return {
        "status": "success",
        "message": "Thank you! Your report has been submitted and will help improve detection accuracy.",
    }


@router.get("/reports")
async def list_reports():
    """List pending user reports."""
    reports = get_pending_reports(limit=100)
    return {"reports": reports, "count": len(reports)}


@router.post("/retrain")
async def trigger_retrain():
    """
    Manually trigger model retraining using accumulated user reports.
    In production this would be a scheduled background job.
    """
    pending = get_pending_reports(limit=10)
    if len(pending) < 5:
        return {
            "status": "skipped",
            "message": f"Need at least 5 new reports to retrain. Currently have {len(pending)}.",
        }

    try:
        # Run retraining script as a background process
        subprocess.Popen(
            ["python", RETRAIN_SCRIPT],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return {
            "status": "started",
            "message": f"Retraining started with {len(pending)} new user reports.",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }
