# AI Autonomous Code Reviewer

## Overview
AI Autonomous Code Reviewer is an advanced AI-powered tool designed to analyze, review, and provide feedback on source code. Using models such as CodeBERT, GPT-4, and Claude AI, this system evaluates code quality, detects potential defects, and suggests improvements. The application offers a FastAPI backend and a Streamlit-based frontend for seamless interaction.

## Features
- **Automated Code Review**: Uses AI models to analyze code and provide feedback.
- **Code Defect Detection**: Identifies potential defects in code using CodeBERT.
- **Linting Integration**: Runs Pylint and ESLint to highlight code style and syntax issues.
- **Multi-Model Support**: Supports GPT-4, Claude, and CodeBERT for analysis.
- **FastAPI Backend**: Provides API endpoints for processing code submissions.
- **Streamlit Frontend**: Simple UI for uploading and analyzing code files.
- **Live Monitoring**: Watches for new code files and triggers AI analysis automatically.
- **Dockerized Deployment**: Runs seamlessly using Docker containers.

## Technologies Used
- **Python**
- **FastAPI**
- **Streamlit**
- **Docker**
- **Hugging Face Transformers (CodeBERT)**
- **Torch**
- **OpenAI API (GPT-4)**
- **Anthropic Claude API**
- **Pylint / ESLint**
- **Watchdog (for live file monitoring)**

## Setup and Installation
Follow these steps to set up and run the AI Code Reviewer.

### 1. Clone the Repository
```sh
git clone https://github.com/your-repository/ai-code-reviewer.git
cd ai-code-reviewer
```

### 2. Install Dependencies (Optional for Local Development)
If running locally, create a virtual environment and install dependencies.
```sh
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the AI Model

Download the model from Hugging Face:
https://huggingface.co/ibrahimpuri/codebertreinforcedmodel
Extract and place the model inside the folder with all the other files and make a folder named "CoderBertModel" directory.

### 4. Set Up Environment Variables
Create a `.env` file and add your API keys:
```
OPENAI_API_KEY=your-openai-api-key
CLAUDE_API_KEY=your-claude-api-key
```

### 5. Run with Docker
Build and run the project using Docker.
```sh
docker-compose up --build
```

Alternatively, run individual containers manually:
```sh
docker build -t ai-code-reviewer .
docker run -p 8000:8000 ai-code-reviewer
```

### 6. Run the Backend (FastAPI)
If running locally without Docker:
```sh
uvicorn fastapi_backend:app --reload
```

### 7. Run the Frontend (Streamlit)
```sh
streamlit run streamlit_app.py
```

### 8. Monitor Code Files for Automatic Review
```sh
python monitordata.py
```

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/review` | `POST` | Upload code and get AI review feedback |

## Usage
1. Open the Streamlit UI.
2. Upload a Python or JavaScript file.
3. Choose an AI tool (GPT-4 or Claude) for analysis.
4. Click "Review Code" to get feedback, defect analysis, and linting results.

## Contributing
We welcome contributions! Feel free to submit issues or pull requests to improve the project.

## License
This project is licensed under the MIT License.
