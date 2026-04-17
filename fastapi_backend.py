import logging

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_code_reviewer_backend import analyze_code

logger = logging.getLogger(__name__)

app = FastAPI(title="CodeLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

VALID_LANGUAGES = {"python", "javascript"}
VALID_AI_TOOLS  = {"gpt", "gpt-4", "claude"}
MAX_FILE_BYTES  = 1 * 1024 * 1024  # 1 MB


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "An internal error occurred."},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


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
