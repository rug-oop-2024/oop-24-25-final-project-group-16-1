import os
from pickle import load
import pandas as pd
import streamlit as st
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import get_metric
from autoop.core.ml.model import CLASSIFICATION_MODELS, get_model
from autoop.core.ml.pipeline import Pipeline

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
    
    if isinstance(selected_pipeline, dict):
        st.write("Here's its content:")

        metadata = selected_pipeline.get("metadata", {})    
        metrics_name = metadata["metrics"]
        reverse_metric_map = {
            "MeanSquaredError": "mean_squared_error",
            "MeanAbsoluteError": "mean_absolute_error",
            "R2Score": "r2_score",
            "Accuracy": "accuracy",
            "Precision": "precision",
            "Recall": "recall",
        }
        metrics = [get_metric(reverse_metric_map.get(f, f)) for f in metrics_name]
        model_name = metadata["model"]
        reverse_model_map = {
            "DecisionTreeModel": "Decision Trees",
            "KNearestNeighbors": "K-Nearest Neighbors",
            "RandomForestModel": "Random Forest",
            "Lasso": "Lasso Regression",
            "LinearRegressionModel": "Linear Regression",
            "MultipleLinearRegression": "Multiple Linear Regression"
        }
        model = get_model(reverse_model_map.get(model_name))
        input_features = metadata["input_features"]
        target_feature = metadata["target_feature"]
        split = metadata["split_ratio"]

        with st.expander("View Pipeline Summary"):
            st.write(f"Metrics: {', '.join(metrics_name)}")
            st.write(f"Model: {model_name}")
            st.write(f"Input Features: {', '.join(input_features)}")
            st.write(f"Target Feature: {target_feature}")
            st.write(f"Split Ratio: {split}")

        input_features_wrapped = [
            Feature(name=f, feature_type=("numerical" if f != target_feature else "categorical"), values=[])
            for f in input_features
        ]
        if model_name in CLASSIFICATION_MODELS:
            target_feature_wrapped = Feature(name=target_feature, feature_type="categorical", values=[])
        else:
            target_feature_wrapped = Feature(name=target_feature, feature_type="numerical", values=[])

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
            asset_path = f"dataset/{dataset_name}.csv"

            new_dataset = Dataset.from_dataframe(
                data=data, name=dataset_name, asset_path=asset_path,
                version="1.0.0"
            )
            selected_pipeline = Pipeline(
                metrics=metrics,
                dataset=new_dataset,
                model=model,
                input_features=input_features_wrapped,
                target_feature=target_feature_wrapped,
                split=split,
            )
            predictions = selected_pipeline.execute()
            with st.expander("View New Training Results"):
                st.write(predictions)

    #if st.button("Delete Pipeline"):
    #    AutoMLSystem.get_instance().registry.delete(f"{selected_name}_1.0.0")
    #    os.remove(selected_file)
    #    st.success(f"Pipeline '{selected_name}' deleted successfully.")
    #    st.rerun()  

else:
    st.write("No saved pipelines found.")
