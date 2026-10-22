"""
PhishGuard Pro - Server Entry Point
Run:
    python run.py
"""
import uvicorn

if __name__ == "__main__":
    print("=" * 56)
    print("  PhishGuard Pro - Starting Server")
    print("  Open browser at: http://localhost:8000")
    print("  API Docs at:     http://localhost:8000/docs")
    print("=" * 56)
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["backend"],
        log_level="info",
    )
