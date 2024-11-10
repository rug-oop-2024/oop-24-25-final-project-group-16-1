import os
from pickle import load
import pandas as pd
import streamlit as st
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import get_metric
from autoop.core.ml.model import CLASSIFICATION_MODELS, get_model
from autoop.core.ml.pipeline import Pipeline

st.set_page_config(page_title="Pipeline Deployment", page_icon="🚀")


def write_helper_text(text: str):
    """
    Display helper text in a lighter color in the Streamlit app.

    Args:
        text (str): The text to display as helper information.
    """
    st.write(f'<p style="color: #888;">{text}', unsafe_allow_html=True)


st.write("# 🚀 Pipeline Deployment")
write_helper_text(
    """
    IMPORTANT INFORMATION:
    <p>
    The uploaded file needs to have a similar structure with the one used
    for the training. If you used nyc_housing.csv as the file
    used for training, then you have to
    upload the nyc_housing_new_predictions.csv</p>
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
        metrics = [
            get_metric(reverse_metric_map.get(f, f)) for f in metrics_name
        ]
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

        st.subheader("Pipeline Summary")
        st.write(f"Metrics: {', '.join(metrics_name)}")
        st.write(f"Model: {model_name}")
        st.write(f"Input Features: {', '.join(input_features)}")
        st.write(f"Target Feature: {target_feature}")
        st.write(f"Split Ratio: {split}")

        if model.name in CLASSIFICATION_MODELS:
            input_features_wrapped = [
                Feature(
                    name=f, feature_type=(
                        "numerical" if f != target_feature else "categorical"
                    ), values=[]
                )
                for f in input_features
            ]
            target_feature_wrapped = Feature(
                name=target_feature, feature_type="categorical", values=[]
            )
        else:
            input_features_wrapped = [
                Feature(name=f, feature_type="numerical", values=[])
                for f in input_features
            ]
            target_feature_wrapped = Feature(
                name=target_feature, feature_type="numerical", values=[]
            )

    st.subheader("Make predictions")
    st.write(
        """
        Upload a CSV file with data similar to the
        pipeline's data to make new predictions.
        """
    )

    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        expected_features = set(input_features)
        uploaded_features = set(data.columns)

        missing_features = expected_features - uploaded_features

        if missing_features:
            missing_features_str = ', '.join(missing_features)
            st.error(
                f"Missing required features in the file:"
                f"{missing_features_str}"
            )
        else:
            asset_path = f"dataset/{selected_name}_addition.csv"
            new_dataset = Dataset.from_dataframe(
                data=data, name=selected_name, asset_path=asset_path,
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

else:
    st.write("No saved pipelines found.")
