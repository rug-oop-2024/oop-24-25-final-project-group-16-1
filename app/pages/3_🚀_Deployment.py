import streamlit as st
from app.core.system import AutoMLSystem

st.set_page_config(page_title="Pipeline Deployment", page_icon="🚀")

st.write("# 🚀 Pipeline Deployment")
st.write(
    """
    Here you can view all saved machine learning pipelines and deploy them if needed.
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
        st.write(f"### Pipeline: {selected.name}")
        st.dataframe(pipeline)

    if st.button("Delete Pipeline"):
        automl.registry.delete(selected.id)
        st.success(f"Pipeline '{selected_name}' deleted successfully.")
        st.rerun()

else:
    st.write("No saved pipelines found.")
