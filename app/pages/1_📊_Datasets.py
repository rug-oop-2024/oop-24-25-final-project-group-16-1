import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

# Initialize the AutoML system
automl = AutoMLSystem.get_instance()

# Fetch the list of datasets from the registry
datasets = automl.registry.list(type="dataset")

# Page title
st.title("Dataset Management")

# Display the list of datasets if available
if datasets:
    st.subheader("Available Datasets")

    # Convert dataset metadata to a DataFrame for better display
    data = [
        {
            "Name": dataset.name,
            "Description": dataset.description,
            "Date Added": dataset.date_added,
        }
        for dataset in datasets
    ]
    df = pd.DataFrame(data)
    st.dataframe(df)

    # Allow the user to select a dataset for viewing or deletion
    selected_dataset_name = st.selectbox("Select a dataset to view or delete:", df["Name"].values)

    # Load the selected dataset and display its contents
    if st.button("View Dataset"):
        selected_dataset = next(dataset for dataset in datasets if dataset.name == selected_dataset_name)
        st.write(f"### Dataset: {selected_dataset.name}")
        st.write(f"Description: {selected_dataset.description}")
        st.write(f"Date Added: {selected_dataset.date_added}")
        st.dataframe(selected_dataset.data)  # Assuming Dataset has a ⁠ data ⁠ attribute

    # Delete dataset
    if st.button("Delete Dataset"):
        automl.registry.delete(selected_dataset_name, type="dataset")
        st.success(f"Dataset '{selected_dataset_name}' deleted successfully.")
        st.experimental_rerun()

else:
    st.write("No datasets available.")

# Section to upload a new dataset
st.subheader("Upload a New Dataset")

uploaded_file = st.file_uploader("Choose a file to upload", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load dataset based on file type
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    dataset_name = st.text_input("Enter dataset name")
    description = st.text_area("Enter dataset description")

    if st.button("Save Dataset"):
        # Save the dataset in the AutoML registry
        new_dataset = Dataset(name=dataset_name, description=description, metadata=data)
        # Example if 'register' is available in ArtifactRegistry
        automl.registry.register(new_dataset)
        st.success(f"Dataset '{dataset_name}' uploaded successfully.")
        st.experimental_rerun()