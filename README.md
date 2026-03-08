MNIST Digit Recognizer App

This project is a Streamlit web app that allows users to draw handwritten digits and predicts the number using a Logistic Regression model trained on the MNIST dataset.

📦 Dataset

We used the MNIST Digit Recognizer dataset from Kaggle:
https://www.kaggle.com/datasets/animatronbot/mnist-digit-recognizer

🧠 About the App

Users can draw digits on a canvas.

The app preprocesses the drawing to match MNIST format.

It predicts the digit with confidence and shows top‑3 guesses.

Includes a clear canvas button and preview of the drawing.

🚀 How It Works

Load and preprocess the digit image.

Flatten and normalize pixel values.

Predict using a trained Logistic Regression model.

Display the result.
