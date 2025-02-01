# README.md

# Tiny LLM App

## Overview

The Tiny LLM App is a lightweight neural network application designed for text classification tasks. This project utilizes the "AG News" dataset from Hugging Face, making it suitable for running on small devices such as mobile phones. The application consists of several components, including model architecture, training logic, and inference capabilities.

## Project Structure

The project is organized as follows:

```
tiny-llm-app
├── data
│   └── dataset.csv          # Dataset used for training the model
├── src
│   ├── model.py             # Defines the neural network architecture
│   ├── train.py             # Contains the training logic for the model
│   ├── inference.py         # Responsible for making predictions with the trained model
│   └── utils.py             # Utility functions for data handling and preprocessing
├── requirements.txt         # Lists the required packages for the project
├── .gitignore               # Specifies files to be ignored by Git
└── README.md                # Documentation for the project
```

## Setup Instructions

To set up the Tiny LLM App, follow these steps:

1. **Create a Virtual Environment**:
   - Use the following command to create a virtual environment:
     ```
     python -m venv venv
     ```
   - Activate the virtual environment:
     - On Windows:
       ```
       venv\Scripts\activate
       ```
     - On macOS/Linux:
       ```
       source venv/bin/activate
       ```

2. **Install Required Packages**:
   - Install the necessary packages listed in `requirements.txt` using pip:
     ```
     pip install -r requirements.txt
     ```

3. **Download the Dataset**:
   - Download the "AG News" dataset from Hugging Face and save it as `dataset.csv` in the `data` directory.

4. **Implement the Neural Network**:
   - Define the architecture of the neural network in `model.py`.

5. **Write the Training Logic**:
   - Implement the training logic in `train.py`, ensuring to preprocess the data and train the model.

6. **Implement Inference Logic**:
   - Write the inference logic in `inference.py` to make predictions using the trained model.

7. **Utilize Utility Functions**:
   - Use `utils.py` for any shared functions needed across the project.

8. **Update Documentation**:
   - Ensure that this README.md file is updated with any additional instructions or information relevant to users.

## Running the Application

- To train the model, run the following command:
  ```
  python src/train.py
  ```

- To make predictions using the trained model, execute:
  ```
  python src/inference.py
  ```

## Contributing

Contributions to the Tiny LLM App are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.