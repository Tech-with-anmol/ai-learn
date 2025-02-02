# Contents of /tiny-llm-app/tiny-llm-app/src/train.py

import pandas as pd  # Importing pandas for data manipulation
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and validation sets
from sklearn.preprocessing import LabelEncoder  # For encoding categorical labels
import tensorflow as tf  # Importing TensorFlow for building and training the neural network
from model import create_model  # Importing the model creation function from model.py
import os  # For handling file paths
from utils import load_dataset  # Import the load_dataset function from utils.py
import pickle  # For saving the tokenizer

# Function to preprocess the dataset
def preprocess_data(df):
    """
    Preprocess the dataset by encoding labels and splitting into features and labels.

    Args:
        df (DataFrame): The DataFrame containing the dataset.

    Returns:
        tuple: A tuple containing the features (X) and labels (y).
    """
    df['label'] = 1  # Assuming 'chosen' is the positive class
    df_rejected = df.copy()
    df_rejected['text'] = df_rejected['rejected']
    df_rejected['label'] = 0  # Assuming 'rejected' is the negative class

    df_chosen = df.copy()
    df_chosen['text'] = df_chosen['chosen']
    df_chosen['label'] = 1

    df_combined = pd.concat([df_chosen, df_rejected])

    X = df_combined['text']
    y = df_combined['label']
    return X, y

# Function to train the model
def train_model(X_train, y_train, X_val, y_val, vocab_size, embedding_dim, num_classes):
    """
    Train the neural network model.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_val (array-like): Validation features.
        y_val (array-like): Validation labels.
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embedding layer.
        num_classes (int): Number of output classes.

    Returns:
        Model: The trained model.
    """
    model = create_model(vocab_size, embedding_dim, num_classes)  # Create the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Compile the model

    # Train the model with the training data
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)  # Adjust epochs and batch size as needed

    return model

# Main function to execute the training process
def main():
    df = load_dataset(os.path.join('data', 'dataset.json'))

    X, y = preprocess_data(df)

    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X)
    X = tokenizer.texts_to_sequences(X)
    X = tf.keras.preprocessing.sequence.pad_sequences(X, padding='post')

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 50
    num_classes = 2  # Binary classification
    model = train_model(X_train, y_train, X_val, y_val, vocab_size, embedding_dim, num_classes)

    model.save('trained_model.keras')  # Save the model in the native Keras format

if __name__ == '__main__':
    main()