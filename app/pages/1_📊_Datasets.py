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
    selected_name = st.selectbox(
        "Select a dataset to view or delete:", dataset_names
    )
    selected = next(
        dataset for dataset in datasets if dataset.name == selected_name
    )

    if st.button("View Dataset"):
        data = selected.read()
        st.write(f"### Dataset: {selected.name}")
        st.dataframe(data)

    if st.button("Delete Dataset"):
        automl.registry.delete(selected.id)
        st.success(f"Dataset '{selected_name}' deleted successfully.")
        st.rerun()
else:
    st.write("No datasets available.")

st.subheader("Upload a New Dataset")
uploaded_file = st.file_uploader(
    "UPLOAD A FILE THAT DOES NOT FINISH WITH _NEW_PREDICTIONS.CSV",
    type=["csv"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    dataset_name = st.text_input("Enter dataset name")

    if dataset_name:
        asset_path = f"dataset/{dataset_name}.csv"

        new_dataset = Dataset.from_dataframe(
            data=data, name=dataset_name, asset_path=asset_path,
            version="1.0.0"
        )

        if st.button("Save Dataset"):
            automl.registry.register(new_dataset)
            st.success(f"Dataset '{dataset_name}' uploaded successfully.")
            st.rerun()
