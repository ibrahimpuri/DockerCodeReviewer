import streamlit as st
from ai_code_reviewer_backend import analyze_code

st.title("AI Autonomous Code Reviewer")
st.sidebar.title("Settings")

# Sidebar options
ai_tool = st.sidebar.radio("Select AI Tool", options=["GPT-4", "Claude"], index=0)
language = st.sidebar.selectbox("Select Programming Language", options=["python", "javascript"])

# File upload
uploaded_file = st.file_uploader("Upload your code file", type=["py", "js"])

if uploaded_file:
    file_content = uploaded_file.read().decode("utf-8")
    st.code(file_content, language=language)

    if st.button("Review Code"):
        st.info("Analyzing code...")
        try:
            results = analyze_code(file_content, language, ai_tool.lower())
            st.subheader("Results")
            st.write(f"**Is Defective**: {'Yes' if results['is_defective'] else 'No'}")
            st.write("**AI Feedback**:")
            st.text(results["feedback"])
            st.write("**Linter Issues**:")
            if results["lint_issues"]:
                st.text("\n".join(results["lint_issues"]))
            else:
                st.text("No linting issues detected.")
        except Exception as e:
            st.error(f"Error analyzing code: {e}")