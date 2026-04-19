import html
import json
import logging
import os

import requests
import streamlit as st
from ai_code_reviewer_backend import analyze_code, reload_model
from database import get_stats, save_training_sample

# In Docker Compose, Streamlit and FastAPI run in separate containers.
# FASTAPI_URL must be the service name (fastapi), not localhost.
# Override to http://localhost:8000 when running outside Docker.
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000")

logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CodeLens · AI Code Reviewer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Base & typography ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---------- Background ---------- */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #12172b 50%, #0d1117 100%);
    min-height: 100vh;
}

/* ---------- Hide Streamlit chrome ---------- */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.15);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* ---------- Sidebar logo area ---------- */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 0 1.5rem 0;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid rgba(99, 102, 241, 0.2);
}
.sidebar-logo-icon {
    font-size: 1.6rem;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 10px;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.sidebar-logo-text {
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.3px;
}
.sidebar-logo-sub {
    font-size: 0.7rem;
    color: #64748b;
    font-weight: 400;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ---------- Sidebar section labels ---------- */
.sidebar-section-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #475569;
    margin: 1.4rem 0 0.5rem 0;
}

/* ---------- Sidebar radio & select overrides ---------- */
[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
    font-size: 0.875rem !important;
}
[data-testid="stSidebar"] .stRadio > div {
    gap: 0.4rem;
}

/* ---------- Hero banner ---------- */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero-badge {
    display: inline-block;
    padding: 4px 14px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 100px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    line-height: 1.15;
    background: linear-gradient(135deg, #f8fafc 0%, #a78bfa 60%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.75rem;
    letter-spacing: -1.5px;
}
.hero-sub {
    font-size: 1.05rem;
    color: #64748b;
    font-weight: 400;
    max-width: 520px;
    margin: 0 auto 2rem;
    line-height: 1.6;
}

/* ---------- Stat pills ---------- */
.stat-row {
    display: flex;
    justify-content: center;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 2.5rem;
}
.stat-pill {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 6px 16px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 100px;
    font-size: 0.8rem;
    color: #94a3b8;
}
.stat-pill span { font-weight: 600; color: #e2e8f0; }

/* ---------- Upload zone ---------- */
.upload-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 0.4rem;
}
[data-testid="stFileUploader"] {
    background: rgba(99, 102, 241, 0.04) !important;
    border: 2px dashed rgba(99, 102, 241, 0.25) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99, 102, 241, 0.5) !important;
}

/* ---------- Code viewer ---------- */
.code-card {
    background: #0d1117;
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 14px;
    overflow: hidden;
    margin-top: 1.5rem;
}
.code-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1rem;
    background: rgba(255,255,255,0.03);
    border-bottom: 1px solid rgba(99, 102, 241, 0.1);
}
.code-card-dots { display: flex; gap: 6px; }
.dot {
    width: 12px; height: 12px; border-radius: 50%;
}
.dot-red   { background: #ff5f57; }
.dot-yellow{ background: #febc2e; }
.dot-green { background: #28c840; }
.code-card-title {
    font-size: 0.75rem;
    color: #475569;
    font-family: 'JetBrains Mono', monospace;
}

/* ---------- Review button ---------- */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2rem !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 24px rgba(99, 102, 241, 0.35) !important;
    margin-top: 1rem !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.5) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ---------- Result cards ---------- */
.result-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}
.result-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
}
.result-card-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #475569;
    margin-bottom: 0.5rem;
}
.result-card-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1;
}
.result-card-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.3rem;
}

/* defective / clean colour variants */
.status-defective { color: #f87171 !important; }
.status-clean     { color: #34d399 !important; }
.card-defective   { border-color: rgba(248, 113, 113, 0.3) !important; background: rgba(248, 113, 113, 0.06) !important; }
.card-clean       { border-color: rgba(52, 211, 153, 0.3) !important; background: rgba(52, 211, 153, 0.06) !important; }

/* ---------- Section headings inside results ---------- */
.section-heading {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #6366f1;
    margin: 2rem 0 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(99, 102, 241, 0.2);
}

/* ---------- AI feedback box ---------- */
.feedback-box {
    background: rgba(99, 102, 241, 0.05);
    border: 1px solid rgba(99, 102, 241, 0.18);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    font-size: 0.9rem;
    color: #cbd5e1;
    line-height: 1.75;
    white-space: pre-wrap;
    font-family: 'Inter', sans-serif;
}

/* ---------- Lint issue rows ---------- */
.lint-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 0.65rem 1rem;
    border-radius: 8px;
    background: rgba(251, 191, 36, 0.05);
    border: 1px solid rgba(251, 191, 36, 0.12);
    margin-bottom: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #fcd34d;
}
.lint-icon { flex-shrink: 0; margin-top: 1px; }
.no-issues {
    padding: 1rem 1.5rem;
    background: rgba(52, 211, 153, 0.06);
    border: 1px solid rgba(52, 211, 153, 0.2);
    border-radius: 10px;
    color: #34d399;
    font-size: 0.875rem;
    font-weight: 500;
}

/* ---------- Truncation warning ---------- */
.trunc-warn {
    padding: 0.75rem 1.25rem;
    background: rgba(251, 191, 36, 0.08);
    border: 1px solid rgba(251, 191, 36, 0.25);
    border-radius: 10px;
    color: #fcd34d;
    font-size: 0.825rem;
    margin-bottom: 1rem;
}

/* ---------- Spinner override ---------- */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* ---------- Divider ---------- */
hr { border-color: rgba(99, 102, 241, 0.12) !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "analysis_meta" not in st.session_state:
    st.session_state.analysis_meta = {}
if "submission_id" not in st.session_state:
    st.session_state.submission_id = None
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False


def _save_feedback(code: str, language: str, predicted: bool, correct: bool) -> None:
    """Derive confirmed label and persist as a training sample."""
    label = int(predicted) if correct else int(not predicted)
    try:
        save_training_sample(code=code, language=language, label=label, source="user_feedback")
        st.session_state.feedback_given = True
        st.rerun()
    except Exception as exc:
        st.error(f"Failed to save feedback: {exc}")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🔍</div>
        <div>
            <div class="sidebar-logo-text">CodeLens</div>
            <div class="sidebar-logo-sub">AI Code Reviewer</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">AI Engine</div>', unsafe_allow_html=True)
    ai_tool = st.radio(
        "AI Engine",
        options=["GPT-4", "Claude"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-section-label">Language</div>', unsafe_allow_html=True)
    language = st.selectbox(
        "Language",
        options=["python", "javascript"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-section-label">API Key Status</div>', unsafe_allow_html=True)
    openai_ok = bool(os.getenv("OPENAI_API_KEY"))
    claude_ok = bool(os.getenv("CLAUDE_API_KEY"))
    st.markdown(
        f'<div style="font-size:0.8rem; color:{"#34d399" if openai_ok else "#f87171"}; margin-bottom:4px;">'
        f'{"✓" if openai_ok else "✗"} OpenAI key {"configured" if openai_ok else "missing"}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:0.8rem; color:{"#34d399" if claude_ok else "#f87171"};">'
        f'{"✓" if claude_ok else "✗"} Anthropic key {"configured" if claude_ok else "missing"}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── DB Stats ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-label">Database Stats</div>', unsafe_allow_html=True)
    try:
        db_stats = get_stats()
        st.markdown(
            f'<div style="font-size:0.8rem; color:#94a3b8; line-height:1.9;">'
            f'📊 Submissions: <strong style="color:#e2e8f0">{db_stats["total_submissions"]}</strong><br>'
            f'🏷️ Training samples: <strong style="color:#e2e8f0">{db_stats["total_training_samples"]}</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            '<div style="font-size:0.75rem; color:#475569;">Stats unavailable</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Retrain button ────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-label">Model Training</div>', unsafe_allow_html=True)
    if st.button("🔁  Retrain Model", use_container_width=True):
        with st.spinner("Fine-tuning in progress…"):
            try:
                resp = requests.post(f"{FASTAPI_URL}/retrain", timeout=600)
                if resp.status_code == 200:
                    train_stats = resp.json().get("training_stats", {})
                    # Reload in this process too — FastAPI and Streamlit each
                    # hold their own model instance loaded from MODEL_WEIGHTS_DIR.
                    reload_model()
                    st.success(
                        f"Retrained on {train_stats.get('samples', '?')} samples. "
                        f"Final loss: {train_stats.get('final_loss', '?')}"
                    )
                elif resp.status_code == 422:
                    st.warning(resp.json().get("detail", "Not enough training samples yet."))
                elif resp.status_code == 409:
                    st.warning("Retraining is already in progress.")
                else:
                    st.error(f"Retrain failed (HTTP {resp.status_code}).")
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach FastAPI backend at {FASTAPI_URL}.")
            except Exception as exc:
                st.error(f"Retrain error: {exc}")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#334155; line-height:1.6; padding-top:0.5rem;">
        Powered by <strong style="color:#6366f1">UniXcoder</strong> for defect
        detection and your chosen AI engine for contextual feedback.
        <br><br>
        Supports <strong style="color:#475569">.py</strong> and
        <strong style="color:#475569">.js</strong> files.
    </div>
    """, unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ AI-Powered Analysis</div>
    <div class="hero-title">Code Review,<br>Reimagined.</div>
    <div class="hero-sub">
        Upload your code and get instant defect detection, AI-generated
        feedback, and linter diagnostics — all in one place.
    </div>
    <div class="stat-row">
        <div class="stat-pill">⚡ <span>UniXcoder</span> defect model</div>
        <div class="stat-pill">🤖 <span>GPT-4</span> · <span>Claude</span></div>
        <div class="stat-pill">🔎 <span>Pylint</span> · <span>ESLint</span></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Upload ─────────────────────────────────────────────────────────────────────
col_upload, col_gap = st.columns([3, 1])
with col_upload:
    st.markdown('<div class="upload-label">Upload File</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your file here or click to browse",
        type=["py", "js"],
        label_visibility="collapsed",
    )

if uploaded_file:
    # Clear stale results when a new file is uploaded
    if st.session_state.analysis_meta.get("file_name") != uploaded_file.name:
        st.session_state.analysis_results = None
        st.session_state.analysis_meta = {}

    # Decode with explicit error handling
    try:
        file_content = uploaded_file.read().decode("utf-8")
    except UnicodeDecodeError:
        st.error(
            "This file could not be read as UTF-8 text. "
            "Only plain-text source files (.py, .js) are supported."
        )
        st.stop()

    # Empty-file guard
    if not file_content.strip():
        st.warning("The uploaded file is empty. Please upload a file with code content.")
        st.stop()

    line_count = file_content.count("\n") + 1
    char_count = len(file_content)

    # Soft size warning (> 1 MB)
    if len(file_content.encode("utf-8")) > 1_000_000:
        st.warning(
            f"This file is larger than 1 MB ({char_count:,} characters). "
            "Analysis may be slow or produce incomplete results."
        )

    # Language / file extension mismatch warning
    ext_map = {"py": "python", "js": "javascript"}
    file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    expected_language = ext_map.get(file_ext)
    if expected_language and expected_language != language:
        st.warning(
            f"File extension `.{file_ext}` suggests **{expected_language}**, "
            f"but the sidebar has **{language}** selected. "
            "Analysis will use the sidebar setting — update it if needed."
        )

    # Code preview card — filename is escaped to prevent XSS
    safe_name = html.escape(uploaded_file.name)
    st.markdown(f"""
    <div class="code-card">
        <div class="code-card-header">
            <div class="code-card-dots">
                <div class="dot dot-red"></div>
                <div class="dot dot-yellow"></div>
                <div class="dot dot-green"></div>
            </div>
            <div class="code-card-title">{safe_name}</div>
            <div style="font-size:0.7rem; color:#334155;">{line_count} lines · {char_count:,} chars</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.code(file_content, language=language)

    if st.button("🔍  Analyse Code", use_container_width=True):
        with st.spinner(
            "Step 1/3 — Defect detection  ·  Step 2/3 — AI feedback  ·  Step 3/3 — Linting…"
        ):
            try:
                st.session_state.analysis_results = analyze_code(
                    file_content, language, ai_tool.lower()
                )
                st.session_state.analysis_meta = {
                    "file_name": uploaded_file.name,
                    "line_count": line_count,
                    "char_count": char_count,
                    "language": language,
                    "ai_tool": ai_tool,
                }
                st.session_state.submission_id = st.session_state.analysis_results.get("submission_id")
                st.session_state.feedback_given = False
            except ValueError as e:
                st.error(f"Input error: {e}")
                st.stop()
            except RuntimeError:
                st.error(
                    "The AI model is not available. "
                    "Check that UniXcoder loaded correctly on startup."
                )
                st.stop()
            except Exception:
                logger.exception("Analysis failed for %s", uploaded_file.name)
                st.error(
                    "Analysis failed due to an unexpected error. "
                    "Verify your API keys are set and network access is available."
                )
                st.stop()

    # ── Results (rendered from session state so they survive sidebar changes) ──
    if st.session_state.analysis_results is not None:
        results     = st.session_state.analysis_results
        is_defective = results.get("is_defective", False)
        lint_issues  = results.get("lint_issues", [])
        truncated    = results.get("truncated", False)
        meta_lines   = st.session_state.analysis_meta.get("line_count", line_count)
        meta_tool    = st.session_state.analysis_meta.get("ai_tool", ai_tool)

        status_class = "status-defective" if is_defective else "status-clean"
        card_class   = "card-defective"   if is_defective else "card-clean"
        status_icon  = "⚠️" if is_defective else "✅"
        status_text  = "Defects Found"    if is_defective else "Looks Clean"

        st.markdown(f"""
        <div class="result-grid">
            <div class="result-card {card_class}">
                <div class="result-card-label">Defect Status</div>
                <div class="result-card-value {status_class}">{status_icon} {status_text}</div>
                <div class="result-card-sub">UniXcoder classification</div>
            </div>
            <div class="result-card">
                <div class="result-card-label">Linter Issues</div>
                <div class="result-card-value">{len(lint_issues)}</div>
                <div class="result-card-sub">{"issues detected" if lint_issues else "no issues found"}</div>
            </div>
            <div class="result-card">
                <div class="result-card-label">AI Engine</div>
                <div class="result-card-value" style="font-size:1.1rem;">{html.escape(meta_tool)}</div>
                <div class="result-card-sub">used for feedback</div>
            </div>
            <div class="result-card">
                <div class="result-card-label">Lines Reviewed</div>
                <div class="result-card-value">{meta_lines:,}</div>
                <div class="result-card-sub">{"⚠ truncated for model" if truncated else "full file analysed"}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if truncated:
            st.markdown("""
            <div class="trunc-warn">
                ⚠ <strong>Note:</strong> Your file exceeds 512 tokens.
                The defect model analysed the first 512 tokens;
                AI feedback covered the full file.
            </div>
            """, unsafe_allow_html=True)

        # ── AI Feedback ────────────────────────────────────────────────────────
        st.markdown('<div class="section-heading">AI Feedback</div>', unsafe_allow_html=True)
        # html.escape() prevents AI-generated HTML/JS from executing in the browser
        safe_feedback = html.escape(results.get("feedback", ""))
        st.markdown(
            f'<div class="feedback-box">{safe_feedback}</div>',
            unsafe_allow_html=True,
        )

        # Copy button: st.code always renders a native clipboard icon
        with st.expander("Copy raw feedback text"):
            st.code(results.get("feedback", ""), language=None)

        # Download results as JSON
        download_payload = json.dumps(
            {
                "file": st.session_state.analysis_meta.get("file_name"),
                "language": st.session_state.analysis_meta.get("language"),
                "ai_tool": st.session_state.analysis_meta.get("ai_tool"),
                "is_defective": is_defective,
                "truncated": truncated,
                "feedback": results.get("feedback", ""),
                "lint_issues": lint_issues,
            },
            indent=2,
        )
        st.download_button(
            label="⬇ Download results as JSON",
            data=download_payload,
            file_name="codelens_results.json",
            mime="application/json",
        )

        # ── Linter Output ──────────────────────────────────────────────────────
        st.markdown('<div class="section-heading">Linter Diagnostics</div>', unsafe_allow_html=True)
        if lint_issues:
            for issue in lint_issues:
                if issue.strip():
                    # html.escape() prevents linter output (which echoes user code) from injecting HTML
                    st.markdown(
                        f'<div class="lint-row"><span class="lint-icon">›</span>'
                        f'{html.escape(issue)}</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                '<div class="no-issues">✓ No linting issues detected.</div>',
                unsafe_allow_html=True,
            )

        # ── Prediction Feedback ────────────────────────────────────────────────
        st.markdown('<div class="section-heading">Prediction Feedback</div>', unsafe_allow_html=True)
        if st.session_state.feedback_given:
            st.markdown(
                '<div style="font-size:0.875rem; color:#34d399; padding:0.75rem 0;">'
                '✓ Thank you — your feedback has been saved as a training sample.</div>',
                unsafe_allow_html=True,
            )
        elif st.session_state.submission_id is not None:
            st.markdown(
                '<div style="font-size:0.8rem; color:#64748b; margin-bottom:0.6rem;">'
                'Was the defect prediction accurate? Your answer improves the model.</div>',
                unsafe_allow_html=True,
            )
            fb_col1, fb_col2 = st.columns(2)
            with fb_col1:
                if st.button("👍  Prediction was correct", use_container_width=True):
                    _save_feedback(
                        code=file_content,
                        language=st.session_state.analysis_meta.get("language", language),
                        predicted=bool(results.get("is_defective", False)),
                        correct=True,
                    )
            with fb_col2:
                if st.button("👎  Prediction was wrong", use_container_width=True):
                    _save_feedback(
                        code=file_content,
                        language=st.session_state.analysis_meta.get("language", language),
                        predicted=bool(results.get("is_defective", False)),
                        correct=False,
                    )

else:
    # Empty state
    st.markdown("""
    <div style="
        text-align:center;
        padding: 4rem 2rem;
        color: #334155;
        border: 2px dashed rgba(99,102,241,0.12);
        border-radius: 18px;
        margin-top: 1rem;
    ">
        <div style="font-size:3rem; margin-bottom:1rem; opacity:0.4;">📂</div>
        <div style="font-size:1rem; font-weight:500; color:#475569;">No file uploaded yet</div>
        <div style="font-size:0.825rem; margin-top:0.4rem; color:#334155;">
            Upload a <code>.py</code> or <code>.js</code> file above to begin.
        </div>
    </div>
    """, unsafe_allow_html=True)
