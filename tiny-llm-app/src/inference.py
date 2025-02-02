# Inference script for the Tiny LLM application

# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import tensorflow as tf  # For loading the trained model and making predictions
from tensorflow.keras.preprocessing.sequence import pad_sequences  # For padding sequences to the same length
from model import create_model  # Import the model creation function from model.py
import pickle  # For loading the tokenizer

# Function to load the trained model
def load_model(model_path):
    """
    Load the trained model from the specified path.

    Args:
        model_path (str): The file path to the saved model.

    Returns:
        model: The loaded Keras model.
    """
    model = tf.keras.models.load_model(model_path)  # Load the model using Keras
    return model  # Return the loaded model

# Function to preprocess input data
def preprocess_input(text, tokenizer, max_length):
    """
    Preprocess the input text for prediction.

    Args:
        text (str): The input text to be processed.
        tokenizer: The tokenizer used for text processing.
        max_length (int): The maximum length for padding sequences.

    Returns:
        numpy.ndarray: The preprocessed and padded input data.
    """
    # Tokenize the input text
    sequences = tokenizer.texts_to_sequences([text])  # Convert text to sequences of integers
    padded_sequences = pad_sequences(sequences, maxlen=max_length)  # Pad sequences to the same length
    return padded_sequences  # Return the padded sequences

# Function to make predictions
def make_prediction(model, input_data):
    """
    Make predictions using the trained model.

    Args:
        model: The trained Keras model.
        input_data (numpy.ndarray): The preprocessed input data.

    Returns:
        numpy.ndarray: The model's predictions.
    """
    predictions = model.predict(input_data)  # Get predictions from the model
    return predictions  # Return the predictions

# Main function to run inference
def main():
    """
    Main function to execute the inference process.
    """
    model_path = 'trained_model.h5'  # Specify the path to the saved model
    tokenizer_path = 'tokenizer.pickle'  # Specify the path to the tokenizer
    max_length = 100  # Define the maximum length for input sequences

    # Load the trained model
    model = load_model(model_path)

    # Load the tokenizer (assuming it's saved as a pickle file)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)  # Load the tokenizer

    # Example input text for prediction
    input_text = "Your input text goes here."  # Replace with actual input text

    # Preprocess the input text
    preprocessed_input = preprocess_input(input_text, tokenizer, max_length)

    # Make predictions
    predictions = make_prediction(model, preprocessed_input)

    # Output the predictions
    print("Predictions:", predictions)  # Print the predictions

# Entry point for the script
if __name__ == "__main__":
    main()  # Run the main function when the script is executed