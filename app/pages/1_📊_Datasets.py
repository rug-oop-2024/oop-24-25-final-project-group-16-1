import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")

st.title("Dataset Management")

if datasets:
    st.subheader("Available Datasets")

    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset to view or delete:", dataset_names)

    if st.button("View Dataset"):
        selected_dataset = next(dataset for dataset in datasets if dataset.name == selected_dataset_name)
        data = selected_dataset.load()
        st.write(f"### Dataset: {selected_dataset.name}")
        st.dataframe(selected_dataset.data)

    if st.button("Delete Dataset"):
        automl.registry.delete(selected_dataset_name)
        st.success(f"Dataset '{selected_dataset_name}' deleted successfully.")
        st.rerun()

else:
    st.write("No datasets available.")

st.subheader("Upload a New Dataset")
uploaded_file = st.file_uploader("Choose a file to upload", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    dataset_name = st.text_input("Enter dataset name")
    asset_path = f"assets/dataset/{dataset_name}"
    new_dataset = Dataset.from_dataframe(name=dataset_name, data=data, asset_path = asset_path, version = "1.0.0")

    if st.button("Save Dataset"):
        automl.registry.register(new_dataset)
        st.success(f"Dataset '{dataset_name}' uploaded successfully.")
        st.rerun()
