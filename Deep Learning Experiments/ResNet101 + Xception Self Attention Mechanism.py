import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet101, Xception
from tensorflow.keras.preprocessing import image_dataset_from_directory
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
data_dir = "F:/Brain Tumor Detection/Dataset/Training"
img_size = (224, 224)
batch_size = 32

train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

# Define the ResNet101 and Xception models
resnet_base = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
xception_base = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of both models to use them as feature extractors
resnet_base.trainable = False
xception_base.trainable = False

# Define a Self-Attention Layer
class SelfAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W_q = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='random_normal')
        self.W_k = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='random_normal')
        self.W_v = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='random_normal')
    
    def call(self, inputs):
        query = tf.matmul(inputs, self.W_q)
        key = tf.matmul(inputs, self.W_k)
        value = tf.matmul(inputs, self.W_v)
        
        scores = tf.matmul(query, key, transpose_b=True)
        scores = tf.nn.softmax(scores / tf.math.sqrt(float(inputs.shape[-1])))
        
        attention_output = tf.matmul(scores, value)
        return attention_output + inputs  # Residual connection

# Get feature maps from both models and apply pooling
def get_feature_maps(input_layer):
    resnet_features = resnet_base(input_layer)
    xception_features = xception_base(input_layer)
    
    # Apply Global Average Pooling to reduce dimensions
    resnet_pooled = layers.GlobalAveragePooling2D()(resnet_features)
    xception_pooled = layers.GlobalAveragePooling2D()(xception_features)
    
    return resnet_pooled, xception_pooled

# Build the combined model
inputs = layers.Input(shape=(224, 224, 3))
resnet_pooled, xception_pooled = get_feature_maps(inputs)

# Concatenate the pooled feature maps from both models
combined_features = layers.Concatenate()([resnet_pooled, xception_pooled])

# Expand dimensions to apply Self-Attention (batch, features, 1) for attention compatibility
expanded_features = layers.Reshape((1, combined_features.shape[-1]))(combined_features)

# Apply self-attention mechanism
attention_output = SelfAttention()(expanded_features)

# Flatten the output back to 1D
attention_flattened = layers.Flatten()(attention_output)

# Add final classification layer
output = layers.Dense(1, activation='sigmoid')(attention_flattened)

# Create the final model
model = models.Model(inputs=inputs, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Save the model
model.save("Residual_Xception_self_attention_mechanism_updated_dataset.h5")


# TESTING THE MODEL

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the path for the test data
data_dir = "F:/Brain Tumor Detection/Dataset/Testing"

# Load the test dataset
batch_size = 32
img_size = (224, 224)
test_ds = image_dataset_from_directory(
    data_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

# Define the SelfAttention layer again for loading purposes
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W_q = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='random_normal')
        self.W_k = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='random_normal')
        self.W_v = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='random_normal')
    
    def call(self, inputs):
        query = tf.matmul(inputs, self.W_q)
        key = tf.matmul(inputs, self.W_k)
        value = tf.matmul(inputs, self.W_v)
        
        scores = tf.matmul(query, key, transpose_b=True)
        scores = tf.nn.softmax(scores / tf.math.sqrt(float(inputs.shape[-1])))
        
        attention_output = tf.matmul(scores, value)
        return attention_output + inputs  # Residual connection

# Load the trained model with custom_objects
model = tf.keras.models.load_model("Residual_Xception_self_attention_mechanism.h5", custom_objects={'SelfAttention': SelfAttention})

# Get the true labels and predictions
true_labels = np.concatenate([y for x, y in test_ds], axis=0)
predictions = model.predict(test_ds)
predicted_labels = (predictions > 0.5).astype("int32").flatten()

# Confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Classification report
class_report = classification_report(true_labels, predicted_labels, target_names=["No Tumor", "Tumor"])

# Print the classification report
print("Classification Report:\n")
print(class_report)

# Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Tumor", "Tumor"], yticklabels=["No Tumor", "Tumor"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Display detailed metrics in a table
metrics = classification_report(true_labels, predicted_labels, target_names=["No Tumor", "Tumor"], output_dict=True)
metrics_df = pd.DataFrame(metrics).transpose()
print("Detailed Performance Table:\n")
print(metrics_df)


# PLOTTING THE Receiver Operating Characteristic (ROC) Curve

import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Get the true labels and predicted probabilities
true_labels = np.concatenate([y for x, y in test_ds], axis=0)
predicted_probs = model.predict(test_ds).ravel()  # Use ravel to flatten probabilities

# Calculate the false positive rate (FPR), true positive rate (TPR), and threshold values
fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
roc_auc = auc(fpr, tpr)  # Calculate the area under the ROC curve

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for reference
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()
