"""
PhishGuard Pro — FastAPI Main Application
==========================================
Dual-layer phishing detection with XAI, Char-CNN, and Active Learning.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import uvicorn

from .routers.analyze import router as analyze_router
from .routers.reports import router as reports_router
from .database.db import init_db

# ─── Initialize DB on startup ─────────────────────────────────────────────────
init_db()

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PhishGuard Pro API",
    description=(
        "Advanced Phishing Detection with Dual-Layer Analysis (URL + Webpage Content), "
        "Explainable AI (SHAP), Char-CNN Typo-Squatting Detection, and "
        "Active Learning from User Feedback."
    ),
    version="1.0.0",
)

# ─── CORS (allow frontend to call API) ────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(analyze_router)
app.include_router(reports_router)

# ─── Serve Frontend Static Files ──────────────────────────────────────────────
FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "PhishGuard Pro"}


# ─── Production entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=port, log_level="info")
