import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained model
try:
    model = tf.keras.models.load_model('mnist.keras')
except (IOError, ImportError):
    gr.Warning("Model file 'mnist.keras' not found. Please run the training script 'cnn-model.py' first to generate the model file.")
    model = None

def predict_digit(image):
    """
    Predicts the digit from a user-drawn image.
    """
    if model is None:
        return "Model not loaded. Please train the model first."

    # Preprocess the image
    img = Image.fromarray(image.astype('uint8'), 'RGB').convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array[..., np.newaxis]
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    
    # Return the prediction as a dictionary of labels and confidences
    confidences = {str(i): float(prediction[0][i]) for i in range(10)}
    return confidences

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(type="numpy", label="Draw a digit here", invert_colors=True, shape=(28, 28)),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="MNIST Digit Recognizer",
    description="Draw a digit from 0 to 9 and see the model's prediction. The model is a CNN trained on the MNIST dataset."
)

if __name__ == "__main__":
    iface.launch()
