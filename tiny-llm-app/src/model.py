# Contents of /tiny-llm-app/tiny-llm-app/src/model.py

import tensorflow as tf  # Import TensorFlow for building the neural network
from tensorflow.keras import layers, models  # Import necessary modules for model creation

class TinyLLMModel(tf.keras.Model):
    """
    This class defines a lightweight neural network model for text classification.
    It inherits from tf.keras.Model to leverage TensorFlow's model functionalities.
    """

    def __init__(self, vocab_size, embedding_dim, num_classes, **kwargs):
        """
        Initializes the TinyLLMModel.

        Parameters:
        vocab_size (int): Size of the vocabulary (number of unique tokens).
        embedding_dim (int): Dimension of the embedding layer.
        num_classes (int): Number of output classes for classification.
        """
        super(TinyLLMModel, self).__init__(**kwargs)  # Call the parent class constructor
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)  # Embedding layer
        self.global_average_pooling = layers.GlobalAveragePooling1D()  # Pooling layer to reduce dimensionality
        self.dense = layers.Dense(16, activation='relu')  # Dense layer with ReLU activation
        self.output_layer = layers.Dense(num_classes, activation='softmax')  # Output layer with softmax activation

    def call(self, inputs):
        """
        Defines the forward pass of the model.

        Parameters:
        inputs: Input data to the model.

        Returns:
        Output of the model after passing through the layers.
        """
        x = self.embedding(inputs)  # Pass inputs through the embedding layer
        x = self.global_average_pooling(x)  # Apply global average pooling
        x = self.dense(x)  # Pass through the dense layer
        return self.output_layer(x)  # Return the output of the model

    def get_config(self):
        """
        Returns the configuration of the model.

        Returns:
        config (dict): Configuration dictionary.
        """
        config = super(TinyLLMModel, self).get_config()
        config.update({
            "vocab_size": self.embedding.input_dim,
            "embedding_dim": self.embedding.output_dim,
            "num_classes": self.output_layer.units
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a model instance from the configuration.

        Parameters:
        config (dict): Configuration dictionary.

        Returns:
        TinyLLMModel: Model instance.
        """
        return cls(**config)

def create_model(vocab_size, embedding_dim, num_classes):
    """
    Function to create and compile the TinyLLMModel.

    Parameters:
    vocab_size (int): Size of the vocabulary.
    embedding_dim (int): Dimension of the embedding layer.
    num_classes (int): Number of output classes.

    Returns:
    model: Compiled TinyLLMModel instance.
    """
    model = TinyLLMModel(vocab_size, embedding_dim, num_classes)  # Instantiate the model
    model.compile(optimizer='adam',  # Use Adam optimizer
                  loss='sparse_categorical_crossentropy',  # Loss function for multi-class classification
                  metrics=['accuracy'])  # Metric to evaluate model performance
    return model  # Return the compiled model