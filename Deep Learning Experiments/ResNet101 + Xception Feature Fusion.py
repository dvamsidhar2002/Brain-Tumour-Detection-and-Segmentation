import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet101, Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Concatenate, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Define paths and parameters
data_path = 'F:/Brain Tumor Detection/Dataset/Training'
batch_size = 32
input_size = (224, 224)  # Adjusted size for ResNet101 and Xception
num_classes = 1  # Binary classification (tumor vs no tumor)

# Data Generators
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    data_path, target_size=input_size,
    class_mode='binary', subset='training', shuffle=True, batch_size=batch_size
)
val_data = datagen.flow_from_directory(
    data_path, target_size=input_size,
    class_mode='binary', subset='validation', shuffle=False, batch_size=batch_size
)

# Define a single input for both branches
input_layer = Input(shape=input_size + (3,))

# Load Pre-trained Models and apply the same input to each
resnet101_base = ResNet101(weights='imagenet', include_top=False, input_tensor=input_layer)
xception_base = Xception(weights='imagenet', include_top=False, input_tensor=input_layer)

# Freeze base models if using transfer learning
resnet101_base.trainable = False
xception_base.trainable = False

# Feature Extraction and Fusion
resnet101_features = GlobalAveragePooling2D()(resnet101_base.output)
xception_features = GlobalAveragePooling2D()(xception_base.output)
combined_features = Concatenate()([resnet101_features, xception_features])

# Classification Layer
output = Dense(1, activation='sigmoid')(combined_features)  # Binary classification with sigmoid

# Create and Compile Model
model_feature_fusion = Model(inputs=input_layer, outputs=output)
model_feature_fusion.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


# Train the model
history = model_feature_fusion.fit(
    train_data,
    validation_data=val_data,
    epochs=20
)

# Save the entire model to a file
model_feature_fusion.save('Residual_Xception_feature_fusion.h5')  # Save as an HDF5 file

# TESTING THE MODEL

# Load the model
model = load_model('F:/Brain Tumor Detection/Deep Learning Experimentaions/Residual_Xception_feature_fusion.h5')

# Prepare test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_data_path = 'F:/Brain Tumor Detection/Dataset/Testing'
input_size = (224, 224)
batch_size = 32

test_data = test_datagen.flow_from_directory(
    test_data_path,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Predict on test data
predictions = model.predict(test_data)
predicted_classes = (predictions > 0.5).astype(int)
true_classes = test_data.classes
class_labels = list(test_data.class_indices.keys())

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Classification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels, output_dict=True)
performance_df = pd.DataFrame(report).transpose()

print("\nDetailed Performance Table:")
print(performance_df)

# PLOTTING Receiver Operating Characteristic (ROC) Curve

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


# ROC Curve
fpr, tpr, thresholds = roc_curve(true_classes, predictions)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

