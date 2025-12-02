import streamlit as st
from pathlib import Path
import json

st.set_page_config(page_title="Dataset Overview", page_icon="📁", layout="wide")

st.title("📁 Dataset Overview")

project_root = Path(__file__).resolve().parents[2]
dataset_dir = project_root / "data" / "raw" / "plantvillage"

st.subheader("📂 Folder Structure")

if dataset_dir.exists():
    st.success(f"Dataset found at: {dataset_dir}")

    for split in ["train", "test", "val"]:
        split_path = dataset_dir / split
        if split_path.exists():
            img_count = len(list(split_path.rglob("*.jpg"))) + len(list(split_path.rglob("*.png")))
            st.write(f"🔸 **{split}** — {img_count} images")
        else:
            st.warning(f"{split} folder missing!")

else:
    st.error(f"❌ Dataset directory not found at: {dataset_dir}")

# -----------------------
# Load dataset statistics
# -----------------------

metrics_path = project_root / "reports" / "results" / "dataset_stats.json"

st.subheader("📊 Dataset Statistics")

if metrics_path.exists():
    try:
        with open(metrics_path, "r") as f:
            stats = json.load(f)

        st.json(stats)

    except Exception as e:
        st.error(f"Could not load dataset_stats.json: {e}")

else:
    st.warning(f"dataset_stats.json not found at: {metrics_path}")
