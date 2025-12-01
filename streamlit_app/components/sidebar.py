# components/sidebar.py

import streamlit as st
from pathlib import Path

def render_sidebar(project_root, check_system_status):
    st.sidebar.title("🌾 Navigation")
    st.sidebar.markdown("---")

    # Page selector
    page = st.sidebar.radio(
        "Go to",
        ("Home", "Disease Prediction", "Model Insights", "Train Model", "About"),
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick actions")
    if st.sidebar.button("Open data folder"):
        st.sidebar.info(f"Project data dir: `{project_root / 'data'}` (open in your editor/OS)")

    st.sidebar.markdown("---")
    st.sidebar.subheader("App features")
    st.sidebar.markdown(
        """
        - Multi-model support (LightGBM / RF / XGBoost)  
        - Image preprocessing & simple explainability  
        - Prediction comparison & history (planned)  
        - Model metadata view
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("System status")
    status = check_system_status()
    st.sidebar.write("Config:", "✅" if status["config_loaded"] else "❌")
    st.sidebar.write("Model:", "✅" if status["model_exists"] else "⚪ (not found)")
    st.sidebar.write("Dataset:", "✅" if status["data_dir_exists"] else "⚪ (optional)")

    st.sidebar.markdown("---")
    with st.sidebar.expander("Need help?"):
        st.sidebar.markdown(
            """
            - Model not found: train locally or copy model to `models/saved_models/`  
            - Wrong predictions: provide clearer images or retrain with more samples  
            - Issues: open an issue in your project's GitHub
            """
        )

    st.sidebar.markdown("---")
    st.sidebar.caption("© 2024 | Built with Streamlit")

    return page
