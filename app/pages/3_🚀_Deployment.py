import pandas as pd
import streamlit as st
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

st.set_page_config(page_title="Pipeline Deployment", page_icon="🚀")

st.write("# 🚀 Pipeline Deployment")
st.write(
    """
    Here you can view all saved machine learning
    pipelines and deploy them if needed.
    Select a pipeline to see its configuration and metrics.
    """
)

automl = AutoMLSystem.get_instance()
pipelines = automl.registry.list(type="pipeline")

if pipelines:
    st.subheader("Available Pipelines")
    pipeline_names = [pipeline.name for pipeline in pipelines]
    selected_name = st.selectbox("Select a Pipeline to view:", pipeline_names)
    selected = next(
        pipeline for pipeline in pipelines if pipeline.name == selected_name
    )

    if st.button("View Pipeline"):
        pipeline = selected.read()
        st.write(f"### You selected this Pipeline: {selected.name}")
        st.write(pipeline)

    st.subheader("Make predictions")
    st.write(
        "Upload a CSV file with data similar as the pipeline"
        "to make new predictions."
    )

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        pipeline._dataset = Dataset.from_dataframe(
            data=data,
            name=selected.name,
            asset_path=selected.asset_path,
            version=selected.version,
        )
        predictions = pipeline.execute()
        st.write(predictions)

    if st.button("Delete Pipeline"):
        automl.registry.delete(selected.id)
        st.success(f"Pipeline '{selected_name}' deleted successfully.")
        st.rerun()

else:
    st.write("No saved pipelines found.")
