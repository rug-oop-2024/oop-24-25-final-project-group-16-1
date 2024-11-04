import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

st.subheader("Select a Dataset")
write_helper_text("Choose a dataset from the list to start designing your machine learning pipeline.")

if datasets:
    data = [
        {
            "Name": dataset.name,
            "Description": dataset.description,
            "Date Added": dataset.date_added,
        }
        for dataset in datasets
    ]
    df = pd.DataFrame(data)
    
    selected_dataset_name = st.selectbox("Available Datasets", df["Name"].values)
    
    if selected_dataset_name:
        selected_dataset = next((d for d in datasets if d.name == selected_dataset_name), None)
        if selected_dataset:
            st.write(f"### Dataset: {selected_dataset.name}")
            st.write(f"Description: {selected_dataset.description}")
            st.write(f"Date Added: {selected_dataset.date_added}")
            st.dataframe(selected_dataset.data.head(100))
            
            features = detect_feature_types(selected_dataset)
            feature_names = [feature.name for feature in features]
            
            st.subheader("Feature Selection")
            write_helper_text("Select the features for your model. Choose multiple input features and one target feature.")
            
            input_features = st.multiselect("Select Input Features", feature_names, default=feature_names[:-1])
            target_feature = st.selectbox("Select Target Feature", feature_names, index=len(feature_names) - 1)
            
            target_feature_type = next((f.feature_type for f in features if f.name == target_feature), None)
            if target_feature_type == "categorical":
                task_type = "Classification"
                available_models = ["Decision Tree Model", "K Nearest Neighbors", "Neural Networks"]
                available_metrics = ["Accuracy", "Precision", "Recall"]
            elif target_feature_type == "numerical" or target_feature_type == "continuous":
                task_type = "Regression"
                available_models = ["Linear Regression", "Lasso Regression", "Multiple Linear Regression"]
                available_metrics = ["Mean Squared Error", "Mean Absolute Error", "R2 Score"]
            else:
                task_type = "Unknown"
                available_models = []
                available_metrics = []
            
            st.write("### Detected Task Type")
            st.write(f"The task type based on the selected target feature is: **{task_type}**")
            
            st.subheader("Model Selection")
            write_helper_text("Select a model compatible with your detected task type.")
            selected_model = st.selectbox("Available Models", available_models)
            
            st.subheader("Select Dataset Split")
            write_helper_text("Choose the proportion of data to use for training and testing.")
            split_ratio = st.slider("Training Data Split (%)", min_value=50, max_value=90, value=80, step=5) / 100.0
            
            st.subheader("Select Metrics")
            write_helper_text("Choose one or more metrics to evaluate your model's performance.")
            selected_metrics = st.multiselect("Available Metrics", available_metrics, default=available_metrics[:1])
            
            st.subheader("Pipeline Summary")
            write_helper_text("Review your pipeline configuration before proceeding.")
            
            st.write("### Selected Configuration")
            st.markdown(f"**Dataset:** {selected_dataset.name}")
            st.markdown(f"**Input Features:** {', '.join(input_features)}")
            st.markdown(f"**Target Feature:** {target_feature}")
            st.markdown(f"**Task Type:** {task_type}")
            st.markdown(f"**Model:** {selected_model}")
            st.markdown(f"**Training Split:** {split_ratio * 100:.0f}%")
            st.markdown(f"**Metrics:** {', '.join(selected_metrics)}")
            
            if st.button("Create Pipeline"):
                st.success("Pipeline created successfully!")
        else:
            st.error("Dataset not found.")
else:
    st.write("No datasets available in the registry.")

