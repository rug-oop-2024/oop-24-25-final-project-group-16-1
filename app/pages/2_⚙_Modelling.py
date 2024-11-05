import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.metric import (
    Accuracy,
    MeanAbsoluteError,
    MeanSquaredError,
    Precision,
    R2Score,
    Recall,
)
from autoop.core.ml.model import (
    CLASSIFICATION_MODELS, REGRESSION_MODELS, get_model
)
from autoop.core.ml.model.classification.decision_tree_classifier import (
    DecisionTreeModel,
)
from autoop.core.ml.model.classification.k_nearest_neighbors import (
    KNearestNeighbors
)
from autoop.core.ml.model.classification.neural_networks import NeuralNetwork
from autoop.core.ml.model.regression.lasso_regression import Lasso
from autoop.core.ml.model.regression.linear_regression import (
    LinearRegressionModel
)
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression,
)
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline to train "
    "a model on a dataset."
)

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")

st.subheader("Select a Dataset")
write_helper_text(
    "Choose a dataset from the list to start designing your machine "
    "learning pipeline."
)

if datasets:
    data = [dataset.name for dataset in datasets]
    df = pd.DataFrame(data)
    selected_dataset_name = st.selectbox(
        "Select a dataset to view or delete:", data
    )
    selected = next(
        dataset for dataset in datasets
        if dataset.name == selected_dataset_name
    )

    if selected_dataset_name:
        selected_dataset = next(
            (d for d in datasets if d.name == selected_dataset_name), None
        )
        if selected_dataset:
            st.write(f"### Dataset: {selected_dataset.name}")
            dataset_df = selected_dataset.read()
            st.dataframe(dataset_df.head(100))

            features = detect_feature_types(selected_dataset)
            feature_names = [feature.name for feature in features]

            st.subheader("Feature Selection")
            write_helper_text(
                "Select the features for your model. Choose multiple input "
                "features and one target feature."
            )

            input_features = st.multiselect(
                "Select Input Features", feature_names,
                default=feature_names[:-1]
            )
            target_feature = st.selectbox(
                "Select Target Feature", feature_names,
                index=len(feature_names) - 1
            )

            target_feature_type = next(
                (
                    f.feature_type for f in features
                    if f.name == target_feature
                ), None
            )
            if target_feature_type == "categorical":
                task_type = "Classification"
                available_models = CLASSIFICATION_MODELS
                available_metrics = ["Accuracy", "Precision", "Recall"]
            elif target_feature_type in {"numerical", "continuous"}:
                task_type = "Regression"
                available_models = REGRESSION_MODELS
                available_metrics = [
                    "Mean Squared Error",
                    "Mean Absolute Error",
                    "R2 Score",
                ]
            else:
                task_type = "Unknown"
                available_models = []
                available_metrics = []

            st.write("### Detected Task Type")
            st.write(
                f"The task type based on the selected target feature is: "
                f"**{task_type}**"
            )

            st.subheader("Model Selection")
            write_helper_text(
                "Select a model compatible with your detected task type."
                )
            selected_model = st.selectbox("Available Models", available_models)

            st.subheader("Select Dataset Split")
            write_helper_text(
                """
                Choose the proportion of data to use for
                training " "and testing.
                """
            )
            split_ratio = (
                st.slider(
                    "Training Data Split (%)",
                    min_value=50,
                    max_value=90,
                    value=80,
                    step=5,
                )
                / 100.0
            )

            st.subheader("Select Metrics")
            write_helper_text(
                """
                Choose one or more metrics to
                evaluate your model's " "performance.
                """
            )
            selected_metrics = st.multiselect(
                "Available Metrics", available_metrics,
                default=available_metrics[:1]
            )

            st.subheader("Pipeline Summary")
            write_helper_text(
                "Review your pipeline configuration before proceeding."
            )

            st.write("### Selected Configuration")
            st.markdown(f"**Dataset:** {selected_dataset.name}")
            st.markdown(f"**Input Features:** {', '.join(input_features)}")
            st.markdown(f"**Target Feature:** {target_feature}")
            st.markdown(f"**Task Type:** {task_type}")
            st.markdown(f"**Model:** {selected_model}")
            st.markdown(f"**Training Split:** {split_ratio * 100:.0f}%")
            st.markdown(f"**Metrics:** {', '.join(selected_metrics)}")

            model_mapping = {
                "Decision Trees": DecisionTreeModel,
                "K Nearest Neighbors": KNearestNeighbors,
                "Neural Networks": NeuralNetwork,
                "Linear Regression": LinearRegressionModel,
                "Lasso Regression": Lasso,
                "Multiple Linear Regression": MultipleLinearRegression,
            }

            if st.button("Train Pipeline"):
                st.write("### Training in Progress")
                model = get_model(selected_model)

                input_f = [f for f in features if f.name in input_features]
                target_f = next(
                    f for f in features if f.name == target_feature
                )
                metric_mapping = {
                    "mean_squared_error": MeanSquaredError,
                    "mean_absolute_error": MeanAbsoluteError,
                    "r2_score": R2Score,
                    "accuracy": Accuracy,
                    "precision": Precision,
                    "recall": Recall,
                }
                metrics = [
                    metric_mapping[name]()
                    for name in selected_metrics
                    if name in metric_mapping
                ]

                pipeline = Pipeline(
                    metrics=metrics,
                    dataset=selected,
                    model=model,
                    input_features=input_f,
                    target_feature=target_f,
                    split=split_ratio,
                )
                pipeline_result = pipeline.execute()

                st.success("Pipeline trained successfully!")
                st.write("### Training Results")
                st.json(
                    {
                        "train_metrics": {
                            metric.__class__.__name__: result
                            for metric, result in pipeline_result[
                                "train_metrics"
                            ]
                        },
                        "test_metrics": {
                            metric.__class__.__name__: result
                            for metric, result in pipeline_result[
                                "test_metrics"
                            ]
                        },
                        "train_predictions": pipeline_result[
                            "train_predictions"
                        ].tolist(),
                        "test_predictions": pipeline_result[
                            "test_predictions"
                        ].tolist(),
                    }
                )
        else:
            st.error("Dataset not found.")
else:
    st.write("No datasets available in the registry.")
