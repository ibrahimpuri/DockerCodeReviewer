import asyncio
import logging

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import ai_code_reviewer_backend as _backend
from ai_code_reviewer_backend import analyze_code, reload_model
from database import get_stats, get_submission, save_training_sample
from trainer import fine_tune

logger = logging.getLogger(__name__)

app = FastAPI(title="CodeLens API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

VALID_LANGUAGES = {"python", "javascript"}
VALID_AI_TOOLS  = {"gpt", "gpt-4", "claude"}
MAX_FILE_BYTES  = 1 * 1024 * 1024  # 1 MB

# Prevents two simultaneous /retrain calls from racing
_retrain_lock = asyncio.Lock()


# ── Request bodies ─────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    submission_id: int
    correct_label: int  # 0 = clean, 1 = defective


# ── Exception handler ──────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "An internal error occurred."},
    )


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Review ─────────────────────────────────────────────────────────────────────

@app.post("/review")
async def review_code(
    file: UploadFile,
    language: str = "python",
    ai_tool: str = "gpt",
):
    if language not in VALID_LANGUAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported language '{language}'. Choose from: {sorted(VALID_LANGUAGES)}",
        )
    if ai_tool not in VALID_AI_TOOLS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported ai_tool '{ai_tool}'. Choose from: {sorted(VALID_AI_TOOLS)}",
        )

    raw = await file.read()
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_FILE_BYTES // 1024} KB size limit.",
        )

    try:
        file_content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=422,
            detail="File is not valid UTF-8. Only plain-text source files are accepted.",
        )

    if not file_content.strip():
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        result = analyze_code(file_content, language, ai_tool)
        return {"success": True, "results": result}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("Unexpected error analysing file: %s", file.filename)
        raise HTTPException(status_code=500, detail="An internal error occurred.")


# ── Feedback ───────────────────────────────────────────────────────────────────

@app.post("/feedback")
async def feedback(body: FeedbackRequest):
    """Save a user-confirmed label for a previous submission as a training sample."""
    if body.correct_label not in (0, 1):
        raise HTTPException(
            status_code=422,
            detail="correct_label must be 0 (clean) or 1 (defective).",
        )

    row = get_submission(body.submission_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Submission {body.submission_id} not found.",
        )

    sample_id = save_training_sample(
        code=row["code"],
        language=row["language"],
        label=body.correct_label,
        source="user_feedback",
    )
    return {"success": True, "training_sample_id": sample_id}


# ── Stats ──────────────────────────────────────────────────────────────────────

@app.get("/stats")
async def stats():
    return {"success": True, "stats": get_stats()}


# ── Retrain ────────────────────────────────────────────────────────────────────

@app.post("/retrain")
async def retrain():
    """
    Fine-tune the model on collected training samples.
    Runs in a thread-pool executor so the event loop is not blocked.
    _retrain_lock prevents two simultaneous calls from racing.
    """
    if _retrain_lock.locked():
        raise HTTPException(
            status_code=409,
            detail="Retraining is already in progress. Please wait.",
        )

    async with _retrain_lock:
        loop = asyncio.get_event_loop()

        def _run():
            result = fine_tune(
                model=_backend.model,
                tokenizer=_backend.tokenizer,
                device=_backend.device,
            )
            reload_model()
            return result

        try:
            result = await loop.run_in_executor(None, _run)
            return {"success": True, "training_stats": result}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception:
            logger.exception("Retraining failed.")
            raise HTTPException(status_code=500, detail="Retraining failed. See server logs.")
