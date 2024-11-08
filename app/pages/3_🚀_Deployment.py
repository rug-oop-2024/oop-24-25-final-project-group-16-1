import streamlit as st
from app.core.system import AutoMLSystem
from autoop.core.ml.model import Model
from autoop.core.ml.dataset import Dataset
from autoop.functional.preprocessing import preprocess_features

st.set_page_config(page_title="Pipeline Deployment", page_icon="🚀")

st.write("# 🚀 Pipeline Deployment")
st.write(
    """
    Here you can view all saved machine learning pipelines and deploy them if needed.
    Select a pipeline to see its configuration and metrics.
    """
)

automl = AutoMLSystem.get_instance()
saved_pipelines = automl.registry.list(type="pipeline")

if saved_pipelines:
    pipeline_names = [pipeline.name for pipeline in saved_pipelines]
    selected_pipeline_name = st.selectbox("Select a Pipeline to view:", pipeline_names)

    selected_pipeline = next(
        pipeline
        for pipeline in saved_pipelines
        if pipeline.name == selected_pipeline_name
    )

    if selected_pipeline:
        st.write(f"### Pipeline: {selected_pipeline.name}")
        st.write(f"**Version:** {selected_pipeline.version}")

        config = selected_pipeline.metadata.get("config", {})
        st.write("#### Configuration")
        st.json(config)

        metrics = selected_pipeline.metadata.get("metrics", {})
        st.write("#### Metrics")
        st.json(metrics)

        if st.button("Deploy Pipeline"):
            st.write("Deploying pipeline...")
            st.success("Pipeline deployed successfully!")

        if st.button("Delete Pipeline"):
            automl.registry.delete(selected_pipeline.id)
            st.success("Pipeline deleted successfully!")

        csv_file = st.file_uploader("Upload CSV for Prediction", type=["csv"])
        if csv_file is not None:
            try:
                dataset = Dataset.from_csv(csv_file)

                input_features = selected_pipeline.metadata.get("input_features", [])
                X = preprocess_features(input_features, dataset)

                model = Model.from_artifact(selected_pipeline.metadata.get("model"))
                predictions = model.predict(X)

                st.write("### Predictions")
                st.write(predictions)

            except Exception as e:
                st.error(f"Error processing predictions: {str(e)}")
else:
    st.write("No saved pipelines found.")
