import streamlit as st
import tensorflow as tf
import pickle

def load_resources():
    # Load the preprocessor
    with open("text_vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)

#     # Load the model
    model = tf.keras.models.load_model("toxicity_model.h5")
    return vec, model

# # Load resources
vec, model = load_resources()

# # Streamlit app layout
st.title("Toxicity Detection App")
st.write("Enter a comment below to check its toxicity level.")

# User input
user_input = st.text_input("Enter your comment: ")

# Define threshold value
threshold = 0.49

# Button to trigger prediction
if st.button("Predict Toxicity"):
    if user_input:
        # Preprocess the input text
        inp = vec([user_input])  # Vec expects a list
        arr = inp.numpy()  # Convert tensor to numpy array

        # Model prediction
        predict = model.predict(arr)

        # Map the predictions to the labels
        labels = [
            "Toxic",
            "Severe_toxic",
            "Obscene",
            "Threat",
            "Insult",
            "Identity_hate"
        ]

        # Evaluate each prediction and apply threshold
        prediction_dict = {}
        for i in range(len(labels)):
            if predict[0][i] > threshold:
                prediction_dict[labels[i]] = "Yes"
            else:
                prediction_dict[labels[i]] = "No"

        # Display results in a nice format
        st.write("### Toxicity Prediction Results:")
        for label, result in prediction_dict.items():
            st.write(f"**{label}:** {result}")
    else:
        st.warning("Please enter a comment to predict its toxicity.")

