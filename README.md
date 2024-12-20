
# Toxicity Detection with LSTM

This project implements a toxicity detection model using Long Short-Term Memory (LSTM) networks to classify text as toxic or non-toxic.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn

You can install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset

The model is trained on a dataset containing text labeled as toxic or non-toxic. Each text is classified with a binary label indicating whether it is toxic (1) or not (0).

## Model

The model uses an LSTM network to analyze the text data. The architecture consists of:

- An Embedding layer
- An LSTM layer
- A Dense layer with a sigmoid activation for binary classification

## Training the Model

1. Preprocess the text data by tokenizing and padding the sequences.
2. Train the model using the training dataset.

```python
model.fit(X_train, y_train, epochs=5, batch_size=64)
```

## Usage

To predict if a text is toxic or not:

```python
def predict_toxicity(text):
    # Preprocess and predict the text
    return 'Toxic' if model.predict(text) >= 0.5 else 'Non-Toxic'
```

## License

This project is licensed under the MIT License.
