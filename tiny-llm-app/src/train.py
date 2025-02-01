# Contents of /tiny-llm-app/tiny-llm-app/src/train.py

import pandas as pd  # Importing pandas for data manipulation
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and validation sets
from sklearn.preprocessing import LabelEncoder  # For encoding categorical labels
import tensorflow as tf  # Importing TensorFlow for building and training the neural network
from model import create_model  # Importing the model creation function from model.py
import os  # For handling file paths

# Function to load the dataset from a CSV file
def load_dataset(file_path):
    """
    Load the dataset from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        DataFrame: A pandas DataFrame containing the dataset.
    """
    return pd.read_csv(file_path)

# Function to preprocess the dataset
def preprocess_data(df):
    """
    Preprocess the dataset by encoding labels and splitting into features and labels.

    Args:
        df (DataFrame): The DataFrame containing the dataset.

    Returns:
        tuple: A tuple containing the features (X) and labels (y).
    """
    # Encoding the labels into numerical format
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])  # Assuming 'label' is the column name for labels
    X = df['text']  # Assuming 'text' is the column name for input text
    y = df['label']
    return X, y

# Function to train the model
def train_model(X_train, y_train, X_val, y_val):
    """
    Train the neural network model.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_val (array-like): Validation features.
        y_val (array-like): Validation labels.

    Returns:
        Model: The trained model.
    """
    model = create_model()  # Create the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Compile the model

    # Train the model with the training data
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)  # Adjust epochs and batch size as needed

    return model

# Main function to execute the training process
def main():
    # Load the dataset
    df = load_dataset(os.path.join('data', 'dataset.csv'))  # Load the dataset from the data directory

    # Preprocess the data
    X, y = preprocess_data(df)  # Get features and labels

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)  # 80-20 split

    # Train the model
    model = train_model(X_train, y_train, X_val, y_val)  # Train the model with the training data

    # Save the trained model
    model.save('trained_model.h5')  # Save the model to a file

if __name__ == '__main__':
    main()  # Execute the main function when the script is run