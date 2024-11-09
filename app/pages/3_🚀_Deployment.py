import os
from pickle import load
import pandas as pd
import streamlit as st
from autoop.core.ml.dataset import Dataset

st.set_page_config(page_title="Pipeline Deployment", page_icon="🚀")

st.write("# 🚀 Pipeline Deployment")
st.write(
    """
    Here you can view all saved machine learning pipelines
    and deploy them if needed.
    Select a pipeline to see its configuration and metrics.
    """
)

pipeline_dir = "./assets/pipelines"
pipeline_files = [f for f in os.listdir(pipeline_dir) if f.endswith(".pkl")]

if pipeline_files:
    st.subheader("Available Pipelines")
    pipeline_names = [os.path.splitext(f)[0] for f in pipeline_files]
    selected_name = st.selectbox("Select a Pipeline to view:", pipeline_names)
    selected_file = os.path.join(pipeline_dir, f"{selected_name}.pkl")

    with open(selected_file, "rb") as f:
        selected_pipeline = load(f)

    with st.expander("View Training Results"):
        st.write(selected_pipeline)

    st.subheader("Make predictions")
    st.write(
        """
        Upload a CSV file with data similar to the
        pipeline's data to make new predictions.
        """
    )

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        dataset_name = st.text_input("Enter dataset name")
        if dataset_name:
            asset_path = f"dataset/{dataset_name}.cvs"

            new_dataset = Dataset.from_dataframe(
                data=data, name=dataset_name, asset_path=asset_path,
                version="1.0.0"
            )

            predictions = selected_pipeline.execute()
            with st.expander("View New Training Results"):
                st.write(predictions)

    if st.button("Delete Pipeline"):
        os.remove(selected_file)
        st.success(f"Pipeline '{selected_name}' deleted successfully.")
        st.rerun()

else:
    st.write("No saved pipelines found.")
