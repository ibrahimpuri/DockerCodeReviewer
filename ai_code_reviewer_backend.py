import os
import subprocess
import tempfile
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

# Suppress warnings from Hugging Face
logging.set_verbosity_error()

# Suppress tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Retrieve API keys securely from the .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# Initialize API clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# Check for MPS (Metal Performance Shaders) support
USE_MPS = torch.backends.mps.is_available()
device = torch.device("mps") if USE_MPS else torch.device("cpu")
print(f"Using device: {device}")

# Load CodeBERT model
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

SUPPORTED_LANGUAGES = {"python", "javascript"}


def gpt4_generate_feedback(code: str) -> str:
    """Generate contextual feedback using GPT-4."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert code reviewer. Provide actionable feedback."},
                {"role": "user", "content": code},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT-4 Error: {e}"


def claude_generate_feedback(code: str) -> str:
    """Generate contextual feedback using Claude."""
    try:
        response = claude_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            system="You are an expert code reviewer. Provide actionable feedback.",
            messages=[
                {"role": "user", "content": code},
            ],
        )
        return response.content[0].text
    except Exception as e:
        return f"Claude Error: {e}"


def run_linter(file_content: str, language: str) -> list:
    """Run the appropriate linter and return issues."""
    suffix = ".py" if language == "python" else ".js"
    issues = []
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
        tmp.write(file_content)
        temp_path = tmp.name
    try:
        if language == "python":
            result = subprocess.run(["pylint", temp_path], capture_output=True, text=True)
            issues = result.stdout.splitlines()
        elif language == "javascript":
            result = subprocess.run(["eslint", temp_path], capture_output=True, text=True)
            issues = result.stdout.splitlines()
    finally:
        os.remove(temp_path)
    return issues


def analyze_code(file_content: str, language: str, ai_tool: str = "gpt") -> dict:
    """Analyze code for defects, generate feedback, and return results."""
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language '{language}'. Supported: {sorted(SUPPORTED_LANGUAGES)}")

    tokenized = tokenizer(file_content, truncation=False)
    token_count = len(tokenized["input_ids"])

    # Warn and truncate if the input exceeds the model's maximum length
    truncated = token_count > 512
    if truncated:
        print(f"Warning: Input tokens exceed model's maximum length ({token_count} > 512). Truncating.")
    inputs = tokenizer(file_content, return_tensors="pt", truncation=True, max_length=512).to(device)

    # Model inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Classification result
    prediction = outputs.logits.argmax(dim=1).item()
    is_defective = prediction == 1

    # AI-generated feedback
    if ai_tool in ("gpt", "gpt-4"):
        feedback = gpt4_generate_feedback(file_content)
    else:
        feedback = claude_generate_feedback(file_content)

    # Run linting
    lint_issues = run_linter(file_content, language)

    return {
        "is_defective": is_defective,
        "truncated": truncated,
        "feedback": feedback,
        "lint_issues": lint_issues,
    }


# Example usage
if __name__ == "__main__":
    example_code = """
    def example_function():
        pass
    """
    language = "python"
    results = analyze_code(example_code, language, ai_tool="gpt")

    print("Results")
    print(f"Is Defective: {'Yes' if results['is_defective'] else 'No'}")
    if results["truncated"]:
        print("Note: Input was truncated to fit model limits.")
    print("\nAI Feedback:\n", results["feedback"])
    print("\nLinter Issues:\n", "\n".join(results["lint_issues"]))
