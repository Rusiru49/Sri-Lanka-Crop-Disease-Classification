import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title="Recommendations", page_icon="⭐", layout="wide")

st.title("⭐ Model Recommendations & Improvements")

project_root = Path(__file__).resolve().parents[2]
rec_path = project_root / "reports" / "results" / "evaluation_metrics.json"

st.subheader("📌 Suggested Improvements")

if rec_path.exists():
    try:
        with open(rec_path, "r") as f:
            data = json.load(f)

        # Check if recommendations exist inside evaluation_metrics.json
        recs = data.get("recommendations", [])

        if recs:
            for item in recs:
                st.markdown(f"### 🔹 {item.get('title', 'Recommendation')}")
                st.write(item.get("description", ""))
        else:
            st.info("No recommendations found inside evaluation_metrics.json")

    except Exception as e:
        st.error(f"Failed to load file: {e}")

else:
    st.error(f"File not found: {rec_path}")
