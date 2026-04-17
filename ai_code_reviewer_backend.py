import logging
import os
import subprocess
import tempfile

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

# Suppress noisy Hugging Face warnings
hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

USE_MPS = torch.backends.mps.is_available()
device = torch.device("mps") if USE_MPS else torch.device("cpu")
logger.info("Using device: %s", device)

model_name = "microsoft/codebert-base"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    logger.info("CodeBERT model loaded successfully.")
except Exception as exc:
    logger.critical(
        "Failed to load CodeBERT model '%s': %s. Defect detection will be unavailable.",
        model_name, exc,
    )
    tokenizer = None
    model = None

SUPPORTED_LANGUAGES = {"python", "javascript"}


def gpt4_generate_feedback(code: str) -> str:
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert code reviewer. Provide actionable feedback."},
                {"role": "user", "content": code},
            ],
            timeout=60,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("GPT-4 feedback error: %s", e)
        return f"GPT-4 Error: {e}"


def claude_generate_feedback(code: str) -> str:
    try:
        response = claude_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            system="You are an expert code reviewer. Provide actionable feedback.",
            messages=[{"role": "user", "content": code}],
            timeout=60,
        )
        return response.content[0].text
    except Exception as e:
        logger.error("Claude feedback error: %s", e)
        return f"Claude Error: {e}"


def run_linter(file_content: str, language: str) -> list:
    suffix = ".py" if language == "python" else ".js"
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
        tmp.write(file_content)
        temp_path = tmp.name
    try:
        if language == "python":
            try:
                result = subprocess.run(
                    ["pylint", temp_path], capture_output=True, text=True, timeout=60
                )
                return result.stdout.splitlines()
            except subprocess.TimeoutExpired:
                return ["Linter timed out after 60 seconds."]
        elif language == "javascript":
            try:
                result = subprocess.run(
                    ["eslint", temp_path], capture_output=True, text=True, timeout=60
                )
                return result.stdout.splitlines()
            except subprocess.TimeoutExpired:
                return ["Linter timed out after 60 seconds."]
    finally:
        os.remove(temp_path)
    return []


def analyze_code(file_content: str, language: str, ai_tool: str = "gpt") -> dict:
    if tokenizer is None or model is None:
        raise RuntimeError(
            "CodeBERT model is not loaded. Check network connectivity and model availability."
        )

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. Supported: {sorted(SUPPORTED_LANGUAGES)}"
        )

    if not file_content or not file_content.strip():
        raise ValueError("File content is empty. Nothing to analyse.")

    tokenized = tokenizer(file_content, truncation=False)
    token_count = len(tokenized["input_ids"])

    truncated = token_count > 512
    if truncated:
        logger.warning("Input tokens exceed model max (%d > 512). Truncating.", token_count)

    inputs = tokenizer(
        file_content, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = outputs.logits.argmax(dim=1).item()
    is_defective = prediction == 1

    if ai_tool in ("gpt", "gpt-4"):
        feedback = gpt4_generate_feedback(file_content)
    else:
        feedback = claude_generate_feedback(file_content)

    lint_issues = run_linter(file_content, language)

    return {
        "is_defective": is_defective,
        "truncated": truncated,
        "feedback": feedback,
        "lint_issues": lint_issues,
    }


if __name__ == "__main__":
    example_code = """
def example_function():
    pass
"""
    results = analyze_code(example_code, "python", ai_tool="gpt")
    print(f"Is Defective: {'Yes' if results['is_defective'] else 'No'}")
    if results["truncated"]:
        print("Note: Input was truncated to fit model limits.")
    print("\nAI Feedback:\n", results["feedback"])
    print("\nLinter Issues:\n", "\n".join(results["lint_issues"]))
