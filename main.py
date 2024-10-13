import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import shap
import random
import os
from imblearn.over_sampling import SMOTE

# Set random seed for reproducibility
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# Load data
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
dataframe = pd.read_csv(data_url)

# Preprocessing
X = dataframe.drop(['name', 'status'], axis=1)
y = dataframe['status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_resampled),
    y=y_train_resampled
)
class_weights_dict = {cls: weight for cls, weight in zip(np.unique(y_train_resampled), class_weights)}
print("\nClass Weights:", class_weights_dict)

# Adjusting the model architecture
def build_model(input_dim):
    model = models.Sequential()

    # Input layer
    model.add(layers.InputLayer(input_shape=(input_dim,)))

    # First hidden layer
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))  
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    # Second hidden layer
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))  
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    # Third hidden layer
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))  
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    # Fourth hidden layer (new)
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

# Instantiate the model
input_dim = X_train_scaled.shape[1]
model = build_model(input_dim)

# Compile model with adjusted learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Reduced learning rate
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]  # Added AUC as an additional metric
)

# Define callbacks
checkpoint_cb = callbacks.ModelCheckpoint(
    'best_model.keras',
    save_best_only=True,
    monitor='val_auc',  # Track AUC
    mode='max',
    verbose=1
)

earlystop_cb = callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True,
    monitor='val_auc',
    mode='max',
    verbose=1
)

lr_schedule = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

# Train the model with class weights
history = model.fit(
    X_train_resampled, y_train_resampled,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=[checkpoint_cb, earlystop_cb, lr_schedule],
    class_weight=class_weights_dict,  # Use class weights
    verbose=1
)

# Evaluate the model on the test set
model.load_weights('best_model.keras')
test_loss, test_accuracy, test_auc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss:  {test_loss:.4f}")
print(f"Test AUC:  {test_auc:.4f}")

# Predictions
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Classification report and confusion matrix
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Parkinson\'s Disease']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:")
print(cm)

# Confusion matrix plot
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Healthy', 'Parkinson\'s Disease'], rotation=45)
plt.yticks(tick_marks, ['Healthy', 'Parkinson\'s Disease'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Annotate confusion matrix
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()

# SHAP explainability
explainer = shap.Explainer(model, X_train_scaled)
shap_values = explainer(X_test_scaled)
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns.tolist())

# ROC-AUC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {test_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
average_precision = average_precision_score(y_test, y_pred_prob)
plt.figure()
plt.plot(recall, precision, label=f'Avg Precision = {average_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

# Save the model
model.save('parkinsons_disease_improved_model.keras')
print("\nModel saved successfully :)")
