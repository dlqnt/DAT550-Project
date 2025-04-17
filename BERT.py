import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import math 

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow_hub as hub
import tensorflow_text as text 


import getpass
import os
import logging

# Set up cache directory for TensorFlow Hub
username = getpass.getuser()
cache_dir = f'/local/home/{username}/tensorflow_hub_cache'
os.environ['TFHUB_CACHE_DIR'] = cache_dir
os.makedirs(cache_dir, exist_ok=True)


# First load CUDA modules with the 'uenv' commands before running the script!
# https://www.ux.uis.no/~trygve-e/GPUCourse.html for commands


# --- GPU and Mixed Precision Setup ---
print("\n--- Setting up GPU and Mixed Precision ---")

# Enable Mixed Precision 
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Mixed precision enabled: {policy.name}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        target_gpu_index = 4 #GPU NUMBER use 'watch gpu-jobs' to see whats in use on gorina
        if target_gpu_index < len(gpus):
            visible_gpus = [gpus[target_gpu_index]]
            tf.config.set_visible_devices(visible_gpus, 'GPU')
            print(f"Attempting to use GPU: {visible_gpus[0].name}")
            tf.config.experimental.set_memory_growth(visible_gpus[0], True)
            print(f"Memory growth enabled for {visible_gpus[0].name}")
        else:
            print(f"GPU index {target_gpu_index} out of range. Available: {len(gpus)}. Check CUDA_VISIBLE_DEVICES or script.")
            if gpus:
                tf.config.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print(f"Falling back to first available GPU: {gpus[0].name}")
            else:
                 raise RuntimeError("No GPUs available after filtering.")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPU found. Running on CPU.")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration ---
print("\n--- Configuration ---")
# Model Selection (BERT Base)
bert_model_name = 'bert_en_uncased_L-12_H-768_A-12' # BERT Base
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
print(f"Using BERT Model: {bert_model_name}")
print(f"Encoder Handle: {tfhub_handle_encoder}")
print(f"Preprocessor Handle: {tfhub_handle_preprocess}")

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 4
INIT_LR = 3e-5
PATIENCE = 2
WEIGHT_DECAY = 0.01 # Standard AdamW weight decay

# Directories
data_dir = "hyperpartisan_data"
bert_dir = f"hyperpartisan_bert_models/{bert_model_name.replace('/', '_')}_tuned_no_official" # Different dir name
os.makedirs(bert_dir, exist_ok=True)
print(f"Model artifacts will be saved in: {bert_dir}")

print(f"TensorFlow version: {tf.__version__}")

# --- Data Loading and Preparation ---
print("\n--- Loading and Preparing Data ---")
articles_df = pd.read_csv(f"{data_dir}/articles_preprocessed.csv")
articles_df['text'] = articles_df['text'].fillna("")

# Data Splitting
train_data, test_data = train_test_split(
    articles_df, test_size=0.2, random_state=42, stratify=articles_df['hyperpartisan']
)
train_data, val_data = train_test_split(
    train_data, test_size=0.2, random_state=42, stratify=train_data['hyperpartisan']
)
print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

# Create TensorFlow Datasets
def create_dataset(data, batch_size=BATCH_SIZE, shuffle=True):
    texts = data['text'].tolist()
    labels = data['hyperpartisan'].astype(int).values
    dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data), seed=42)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_data)
val_dataset = create_dataset(val_data, shuffle=False)
test_dataset = create_dataset(test_data, shuffle=False)
print("TF Datasets created.")

# --- Model Building ---
print("\n--- Building BERT Model ---")

def build_bert_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier', dtype=tf.float32)(net)
    return tf.keras.Model(inputs=text_input, outputs=net)

bert_model = build_bert_model()
bert_model.summary()
print("BERT model built.")


# --- Optimizer and Compilation ---
print("\n--- Compiling Model (using Keras AdamW + PolynomialDecay Schedule) ---")
# Calculate training steps
# Use math.ceil to ensure all data is seen if the dataset size isn't perfectly divisible by batch size
steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE)
num_train_steps = steps_per_epoch * EPOCHS
# num_warmup_steps = int(0.1 * num_train_steps) # Warmup steps are not explicitly used in PolynomialDecay schedule

# Create a learning rate schedule (Linear Decay)
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=INIT_LR,
    decay_steps=num_train_steps, # Decay over the course of training
    end_learning_rate=0.0,       # Decay to zero
    power=1.0                    # power=1.0 is linear decay
)
print(f"Using PolynomialDecay schedule: init_lr={INIT_LR}, decay_steps={num_train_steps}, end_lr=0.0")

# Create the AdamW optimizer with the schedule
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=WEIGHT_DECAY  # Apply weight decay
)
print(f"Optimizer: AdamW with weight_decay={WEIGHT_DECAY}")


# Define metrics
metrics = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

bert_model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=metrics
)
print("Model compiled.")

# --- Callbacks ---
print("\n--- Setting up Callbacks ---")
checkpoint_path = os.path.join(bert_dir, 'bert_model_checkpoint_best.weights.h5')
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_auc',
    mode='max',
    save_best_only=True,
    save_weights_only=True
)

early_stopping = EarlyStopping(
    monitor='val_auc',
    mode='max',
    patience=PATIENCE,
    restore_best_weights=True
)
print(f"Callbacks: ModelCheckpoint (monitor='val_auc', mode='max'), EarlyStopping (monitor='val_auc', mode='max', patience={PATIENCE})")


# --- Training ---
print("\n--- Training BERT Model ---")
print(f"Training for up to {EPOCHS} epochs with Batch Size {BATCH_SIZE}")
# print(f"Warmup Steps (info only): {num_warmup_steps}, Total Steps: {num_train_steps}") # Note: Warmup not explicitly in schedule
history = bert_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[model_checkpoint, early_stopping]
)

# --- Evaluation ---
print("\n--- Evaluating BERT Model on Test Data ---")
results = bert_model.evaluate(test_dataset, verbose=0)
test_loss, test_accuracy, test_auc, test_precision, test_recall = results[0], results[1], results[2], results[3], results[4]
test_f1 = 0.0
if test_precision + test_recall > 0:
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

# --- Detailed Report and Confusion Matrix ---
print("\n--- Generating Detailed Test Report ---")
y_true_list = []
y_pred_prob_list = []
print("Predicting on test set...")
for texts, labels in test_dataset:
    preds = bert_model.predict(texts, verbose=0)
    y_true_list.extend(labels.numpy())
    y_pred_prob_list.extend(preds.flatten())

y_true = np.array(y_true_list)
y_pred_prob = np.array(y_pred_prob_list)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Not Hyperpartisan (0)', 'Hyperpartisan (1)']))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(f"{'':<25} {'Predicted Not Hyp (0)':<25} {'Predicted Hyp (1)':<25}")
print(f"{'True Not Hyp (0)':<25} {cm[0][0]:<25d} {cm[0][1]:<25d}")
print(f"{'True Hyp (1)':<25} {cm[1][0]:<25d} {cm[1][1]:<25d}")

print(f"\nBest model weights were saved during training to: {checkpoint_path}")

# --- Training History Summary ---
print("\n--- Training History Summary ---")
if history and history.history:
    for epoch in range(len(history.history['loss'])):
        print(f"Epoch {epoch+1}: "
              f"loss={history.history['loss'][epoch]:.4f}, "
              f"accuracy={history.history['accuracy'][epoch]:.4f}, "
              f"auc={history.history['auc'][epoch]:.4f}, "
              f"val_loss={history.history['val_loss'][epoch]:.4f}, "
              f"val_accuracy={history.history['val_accuracy'][epoch]:.4f}, "
              f"val_auc={history.history['val_auc'][epoch]:.4f}")
else:
    print("No training history recorded (possibly loaded from checkpoint).")

print("\nBERT implementation complete!")