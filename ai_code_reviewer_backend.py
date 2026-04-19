import logging
import os
import subprocess
import tempfile
import threading

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

from database import init_db, save_submission

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

def _detect_device() -> torch.device:
    """
    Detect the best available compute device.

    Priority: CUDA (NVIDIA / AMD ROCm) > MPS (Apple Silicon) > XPU (Intel) > CPU

    Notes:
    - AMD ROCm builds of PyTorch present their device as "cuda", so
      torch.cuda.is_available() returns True on both NVIDIA and AMD GPUs.
    - MPS is only available when running natively on macOS; it is never
      accessible inside a Docker container.
    - Intel XPU requires intel-extension-for-pytorch to be installed.
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info("GPU detected: %s (CUDA/ROCm)", name)
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        logger.info("GPU detected: Apple Silicon (MPS)")
        return torch.device("mps")

    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            name = torch.xpu.get_device_name(0)
            logger.info("GPU detected: %s (Intel XPU)", name)
            return torch.device("xpu")
    except ImportError:
        pass

    logger.info("No GPU detected — running on CPU")
    return torch.device("cpu")


device = _detect_device()

MODEL_NAME        = "microsoft/unixcoder-base"
MODEL_WEIGHTS_DIR = os.getenv("MODEL_WEIGHTS_DIR", "./data/model_weights")

# RLock allows the same thread to re-acquire (e.g. reload_model called while lock is held)
_model_lock = threading.RLock()


def _load_model():
    """
    Load tokenizer + model. Uses local fine-tuned weights if they exist,
    otherwise downloads from HuggingFace.
    Returns (tokenizer, model) or (None, None) on failure.
    """
    local_config = os.path.join(MODEL_WEIGHTS_DIR, "config.json")
    source = MODEL_WEIGHTS_DIR if os.path.isfile(local_config) else MODEL_NAME
    logger.info("Loading model from: %s", source)
    try:
        tok = AutoTokenizer.from_pretrained(source)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            source, num_labels=2
        ).to(device)
        mdl.eval()
        logger.info("UniXcoder model loaded successfully.")
        return tok, mdl
    except Exception as exc:
        logger.critical(
            "Failed to load model from '%s': %s. Defect detection unavailable.",
            source, exc,
        )
        return None, None


tokenizer, model = _load_model()


def reload_model() -> None:
    """
    Hot-swap global model + tokenizer with freshly saved weights.
    Acquires _model_lock so any in-flight forward pass finishes first.
    Called by fastapi_backend.py after fine-tuning completes.
    """
    global tokenizer, model
    with _model_lock:
        logger.info("Reloading model weights...")
        tokenizer, model = _load_model()
        logger.info("Model reload complete.")


# Initialise DB tables (safe to call multiple times)
init_db()

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
            "UniXcoder model is not loaded. Check network connectivity and model availability."
        )

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. Supported: {sorted(SUPPORTED_LANGUAGES)}"
        )

    if not file_content or not file_content.strip():
        raise ValueError("File content is empty. Nothing to analyse.")

    tokenized   = tokenizer(file_content, truncation=False)
    token_count = len(tokenized["input_ids"])
    truncated   = token_count > 512
    if truncated:
        logger.warning("Input tokens exceed model max (%d > 512). Truncating.", token_count)

    inputs = tokenizer(
        file_content, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    # Hold the lock only for the forward pass so the model can be reloaded between requests
    with _model_lock:
        with torch.no_grad():
            outputs = model(**inputs)

    prediction   = outputs.logits.argmax(dim=1).item()
    is_defective = prediction == 1

    if ai_tool in ("gpt", "gpt-4"):
        feedback = gpt4_generate_feedback(file_content)
    else:
        feedback = claude_generate_feedback(file_content)

    lint_issues = run_linter(file_content, language)

    submission_id = save_submission(
        code=file_content,
        language=language,
        ai_tool=ai_tool,
        is_defective=is_defective,
    )

    return {
        "submission_id": submission_id,
        "is_defective":  is_defective,
        "truncated":     truncated,
        "feedback":      feedback,
        "lint_issues":   lint_issues,
    }


if __name__ == "__main__":
    example_code = """
def example_function():
    pass
"""
    results = analyze_code(example_code, "python", ai_tool="gpt")
    print(f"Submission ID : {results['submission_id']}")
    print(f"Is Defective  : {'Yes' if results['is_defective'] else 'No'}")
    if results["truncated"]:
        print("Note: Input was truncated to fit model limits.")
    print("\nAI Feedback:\n", results["feedback"])
    print("\nLinter Issues:\n", "\n".join(results["lint_issues"]))
