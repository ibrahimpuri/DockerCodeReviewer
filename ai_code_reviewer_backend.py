import os
import subprocess
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, logging
import openai
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

# Initialize APIs
openai.api_key = OPENAI_API_KEY
claude_client = anthropic.Client(api_key=CLAUDE_API_KEY)

# Check for MPS (Metal Performance Shaders) support
USE_MPS = torch.backends.mps.is_available()
device = torch.device("mps") if USE_MPS else torch.device("cpu")
print(f"Using device: {device}")

# Load CodeBERT model
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

def gpt4_generate_feedback(code: str) -> str:
    """Generate contextual feedback using GPT-4."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert code reviewer. Provide actionable feedback."},
                {"role": "user", "content": code},
            ],
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"GPT-4 Error: {e}"

def claude_generate_feedback(code: str) -> str:
    """Generate contextual feedback using Claude."""
    try:
        response = claude_client.chat(
            model="claude-2",
            messages=[
                {"role": "system", "content": "You are an expert code reviewer. Provide actionable feedback."},
                {"role": "user", "content": code},
            ],
            max_tokens_to_sample=1000,
        )
        return response["completion"]
    except Exception as e:
        return f"Claude Error: {e}"

def run_linter(file_content: str, language: str) -> list:
    """Run the appropriate linter and return issues."""
    temp_path = "temp_code_file.py" if language == "python" else "temp_code_file.js"
    with open(temp_path, "w") as f:
        f.write(file_content)
    issues = []
    if language == "python":
        result = subprocess.run(["pylint", temp_path], capture_output=True, text=True)
        issues = result.stdout.splitlines()
    elif language == "javascript":
        result = subprocess.run(["eslint", temp_path], capture_output=True, text=True)
        issues = result.stdout.splitlines()
    os.remove(temp_path)
    return issues

def split_into_chunks(text, max_length=512):
    """Splits long input into smaller chunks for processing."""
    tokenized = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"]
    return [tokenized[0][i:i + max_length] for i in range(0, len(tokenized[0]), max_length)]

def analyze_code(file_content: str, language: str, ai_tool: str = "gpt") -> dict:
    """Analyze code for defects, generate feedback, and return results."""
    tokenized = tokenizer(file_content, truncation=False)
    token_count = len(tokenized["input_ids"])

    # Warn and truncate if the input exceeds the model's maximum length
    if token_count > 512:
        print(f"Warning: Input tokens exceed model's maximum length ({token_count} > 512). Truncating.")
    inputs = tokenizer(file_content, return_tensors="pt", truncation=True, max_length=512).to(device)

    # Model inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Classification result
    prediction = outputs.logits.argmax(dim=1).item()
    is_defective = prediction == 1

    # AI-generated feedback
    feedback = gpt4_generate_feedback(file_content) if ai_tool == "gpt" else claude_generate_feedback(file_content)

    # Run linting
    lint_issues = run_linter(file_content, language)

    return {
        "is_defective": is_defective,
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
    print("\nAI Feedback:\n", results["feedback"])
    print("\nLinter Issues:\n", "\n".join(results["lint_issues"]))