from fastapi import FastAPI, UploadFile, HTTPException
from ai_code_reviewer_backend import analyze_code

app = FastAPI()

@app.post("/review")
async def review_code(file: UploadFile, language: str = "python", ai_tool: str = "gpt"):
    file_content = (await file.read()).decode("utf-8")
    try:
        result = analyze_code(file_content, language, ai_tool)
        return {"success": True, "results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))