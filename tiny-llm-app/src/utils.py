# utils.py

# This file contains utility functions that are used across the other files in the project.
# These functions include data preprocessing, loading the dataset, and other helper functions.

import os  # For handling file paths and extensions
import pandas as pd  # For data manipulation

def load_dataset(file_path):
    """
    Load a dataset from a CSV or JSON file.

    Parameters:
        file_path (str): The path to the dataset file (CSV or JSON).

    Returns:
        DataFrame: A pandas DataFrame containing the dataset.
    """
    _, ext = os.path.splitext(file_path)  # Get the file extension
    if ext.lower() == '.json':
        # For a JSON file that contains an array of objects (your dataset format),
        # use pd.read_json with default settings.
        return pd.read_json(file_path)
    elif ext.lower() == '.csv':
        # For CSV files, use the Pandas CSV reader.
        return pd.read_csv(file_path)
    else:
        # Raise an error if the file format is unsupported.
        raise ValueError("Unsupported file format. Only CSV and JSON are supported.")

def load_ai_dataset(file_path):
    """
    Load an AI dataset from a JSON file and verify the expected columns.
    
    Expected dataset format:
    [
        {
            "prompt": "The input prompt text",
            "chosen": "The accepted output text",
            "rejected": "The rejected output text"
        },
        ...
    ]
    
    Parameters:
        file_path (str): The path to the dataset file.

    Returns:
        DataFrame: A pandas DataFrame with columns 'prompt', 'chosen', and 'rejected'.
    """
    # Load the dataset using the previously defined function
    df = load_dataset(file_path)
    
    # Define the expected columns
    expected_cols = {"prompt", "chosen", "rejected"}
    
    # Verify that all expected columns exist in the dataset.
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"Dataset must contain columns: {expected_cols}")
    
    # Return the loaded DataFrame
    return df

def preprocess_text(text):
    """
    Preprocess the input text for the model.

    Parameters:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    # Example preprocessing: convert text to lowercase.
    # More steps (like removing punctuation or stopwords) can be added as needed.
    text = text.lower()
    return text

def split_data(data, test_size=0.2):
    """
    Split the dataset into training and testing sets.

    Parameters:
    data (DataFrame): The DataFrame containing the dataset.
    test_size (float): The proportion of the dataset to include in the test split.

    Returns:
    tuple: A tuple containing the training and testing DataFrames.
    """
    # Calculate the number of test samples
    test_count = int(len(data) * test_size)
    # Split the data into training and testing sets
    train_data = data[:-test_count]
    test_data = data[-test_count:]
    return train_data, test_data

def get_labels(data, label_column):
    """
    Extract labels from the dataset.

    Parameters:
    data (DataFrame): The DataFrame containing the dataset.
    label_column (str): The name of the column containing the labels.

    Returns:
    Series: A pandas Series containing the labels.
    """
    # Return the specified label column from the DataFrame
    return data[label_column]