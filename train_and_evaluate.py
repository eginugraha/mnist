# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import keras_tuner as kt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# --- 1. Data Loading and Preparation ---
# Load the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits and their labels.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Add a channel dimension to the images. CNNs expect input in the format (height, width, channels).
# For grayscale images, the channel dimension is 1.
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# --- 2. Preprocessing --- 
# Define a function to preprocess the images.
def preprocess(image, label):
    # Cast the image data type to float32 and normalize pixel values from [0, 255] to [0, 1].
    # Normalization helps the model converge faster.
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Set the batch size for training. A smaller batch size uses less memory.
batch_size = 32

# Create a TensorFlow Dataset for the training data.
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# Apply preprocessing, shuffle the data to ensure randomness, batch it, and prefetch for performance.
ds_train = ds_train.map(preprocess).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Create a TensorFlow Dataset for the test data.
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# Apply the same preprocessing and batching. Shuffling is not needed for the test set.
ds_test = ds_test.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- 3. Model Building ---
# Define a function to build the CNN model. `hp` (hyperparameters) is an object from Keras Tuner.
def build_model(hp):
    model = models.Sequential()
    
    # Add an Input layer to define the model's input shape, which is the recommended practice.
    model.add(tf.keras.Input(shape=(28, 28, 1)))
    
    # First Convolutional Layer: Extracts features from the input image.
    model.add(layers.Conv2D(
        # Tune the number of filters. More filters can capture more complex patterns.
        filters=hp.Choice('conv1_filter', [16, 32, 64]),
        kernel_size=3,       # 3x3 kernel size is standard.
        activation='relu'   # ReLU activation function introduces non-linearity.
    ))
    # Max Pooling Layer: Downsamples the feature map to reduce dimensionality.
    model.add(layers.MaxPooling2D())
    
    # Second Convolutional Layer: Extracts more abstract features.
    model.add(layers.Conv2D(
        # Tune the number of filters for the second layer.
        filters=hp.Choice('conv2_filter', [16, 32]),
        kernel_size=3,
        activation='relu'
    ))
    model.add(layers.MaxPooling2D())
    
    # Flatten Layer: Converts the 2D feature maps into a 1D vector to feed into the dense layers.
    model.add(layers.Flatten())
    
    # Dense Layer: A fully connected layer for classification.
    model.add(layers.Dense(
        # Tune the number of units (neurons) in the dense layer.
        units=hp.Choice('dense_units', [32, 64]),
        activation='relu'
    ))
    # Dropout Layer: Regularization technique to prevent overfitting by randomly setting a fraction of inputs to 0.
    model.add(layers.Dropout(0.3))
    
    # Output Layer: A dense layer with 10 units (one for each digit 0-9) and softmax activation.
    # Softmax outputs a probability distribution over the classes.
    model.add(layers.Dense(10, activation='softmax'))
    
    # Compile the model with an optimizer, loss function, and metrics.
    model.compile(
        # Adam optimizer is a good default choice. Tune the learning rate.
        optimizer=tf.keras.optimizers.Adam(hp.Choice('lr', [1e-3, 5e-4])),
        # SparseCategoricalCrossentropy is used for multi-class classification with integer labels.
        loss='sparse_categorical_crossentropy',
        # Track accuracy during training.
        metrics=['accuracy']
    )
    return model

# --- 4. Hyperparameter Tuning --- 
# Initialize the Keras Tuner with the Hyperband algorithm.
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',  # The metric to optimize.
    max_epochs=5,              # Maximum number of epochs to train one model.
    factor=3,                  # Reduction factor for the number of models and epochs per bracket.
    directory='tunning_results', # Directory to store the tuning results.
    project_name='mnist'
)

# Define an EarlyStopping callback to stop training if the validation loss doesn't improve.
# This prevents overfitting and saves time.
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

# Start the hyperparameter search process.
tuner.search(ds_train, validation_data=ds_test, epochs=5, callbacks=[stop_early])

# --- 5. Save the Best Model ---
# Retrieve the best model found by the tuner.
best_model = tuner.get_best_models(1)[0]
# Save the best model to a file for later use in the application.
best_model.save('mnist.keras')

# --- 6. Evaluate the Best Model ---
# Retrieve the best hyperparameters.
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the best hyperparameters and train it on the full training data.
model = build_model(best_hps)
history = model.fit(ds_train, epochs=10, validation_data=ds_test) # Train for a few more epochs

# Save the final trained model
model.save('mnist.keras')

# --- 7. Generate and Save Evaluation Results ---

# --- Model Summary ---
with open('evaluation_results/model_summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# --- Accuracy and Loss Plots ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('evaluation_results/accuracy_plot.png')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('evaluation_results/loss_plot.png')
plt.close()

# --- Confusion Matrix and Classification Report ---
# Get predictions on the test set
y_pred_probs = model.predict(ds_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.concatenate([y for x, y in ds_test], axis=0)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('evaluation_results/confusion_matrix.png')
plt.close()

# Classification Report
report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)])
with open('evaluation_results/classification_report.txt', 'w') as f:
    f.write(report)

print("\nEvaluation results (summary, plots, confusion matrix, and report) have been saved to the 'evaluation_results' directory.")