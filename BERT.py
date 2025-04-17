# BERT for Hyperpartisan News Detection (Metrics Only Version)
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow_hub as hub
import tensorflow_text as text
#from official.nlp import optimization  # Efficient learning rate scheduler
# GPU configuration
import getpass
import os

# Set up cache directory for TensorFlow Hub
username = getpass.getuser()
cache_dir = f'/local/home/{username}/tensorflow_hub_cache'
os.environ['TFHUB_CACHE_DIR'] = cache_dir
os.makedirs(cache_dir, exist_ok=True)

# GPU setup using CUDA modules (important for Gorina servers)
# First load CUDA modules with the 'uenv' commands before running the script!
# https://www.ux.uis.no/~trygve-e/GPUCourse.html

# Select specific GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set GPU #4 
        gpu_number = 4  
        
        if gpu_number < len(gpus):
            tf.config.set_visible_devices(gpus[gpu_number], 'GPU')
            print(f"Using GPU: {gpus[gpu_number]}")
        else:
            print(f"GPU {gpu_number} not found. Available GPUs: {len(gpus)}")
            print(f"Using first available GPU instead.")
            tf.config.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        # Runtime errors occur if this is called after GPU initialization
        print(f"GPU selection error: {e}")
    
    # Limit GPU memory growth (helps prevent OOM errors)
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth enabled for {gpu}")
        except RuntimeError as e:
            print(f"Memory growth setting error: {e}")
else:
    print("No GPU found. Running on CPU.")
# Suppress TensorFlow warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create directories for saving models
bert_dir = "hyperpartisan_bert_models"
os.makedirs(bert_dir, exist_ok=True)

print("TensorFlow version:", tf.__version__)

# Load the preprocessed data
print("\nLoading preprocessed data...")
data_dir = "hyperpartisan_data"
articles_df = pd.read_csv(f"{data_dir}/articles_preprocessed.csv")

# Handle missing values
articles_df['text'] = articles_df['text'].fillna("")

# Text length statistics
text_lengths = articles_df['text'].str.split().str.len()
print(f"Text length statistics (word count):")
print(f"  Mean: {text_lengths.mean():.1f}")
print(f"  Median: {text_lengths.median():.1f}")
print(f"  Max: {text_lengths.max()}")

# Split the data
print("\nSplitting data into train/val/test sets...")
train_data, test_data = train_test_split(
    articles_df, test_size=0.2, random_state=42, stratify=articles_df['hyperpartisan']
)

train_data, val_data = train_test_split(
    train_data, test_size=0.2, random_state=42, stratify=train_data['hyperpartisan']
)

print(f"Training set: {len(train_data)} articles")
print(f"Validation set: {len(val_data)} articles")
print(f"Test set: {len(test_data)} articles")
print(f"Class distribution in training set: \n{train_data['hyperpartisan'].value_counts()}")

# Load BERT model
print("\nLoading BERT model from TensorFlow Hub...")

bert_model_name = 'bert_en_uncased_L-12_H-768_A-12'
map_name_to_handle = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4', # Check for latest version
}
map_model_to_preprocess = {
     'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3', 
}

tfhub_handle_encoder = map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

print(f"BERT model selected: {tfhub_handle_encoder}")
print(f"Preprocessing model: {tfhub_handle_preprocess}")

# Create preprocessing layer
bert_preprocess = hub.load(tfhub_handle_preprocess)
bert_encoder = hub.load(tfhub_handle_encoder)

# Create BERT model
def build_bert_model():
    # Text inputs
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    
    # BERT preprocessing and encoder
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    
    # Connect the layers
    encoder_inputs = preprocessing_layer(text_input)
    outputs = encoder(encoder_inputs)
    
    # Use pooled_output for classification
    net = outputs['pooled_output']
    
    # Add dropout
    net = tf.keras.layers.Dropout(0.1)(net)
    
    # Classification layer
    net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)
    
    # Build model
    model = tf.keras.Model(inputs=text_input, outputs=net)
    return model


print("\nBuilding BERT model...")
bert_model = build_bert_model()
print("BERT model built successfully")

# Prepare training datasets
print("\nPreparing training datasets...")
BATCH_SIZE = 16
EPOCHS = 5

# Create TensorFlow datasets
def create_dataset(data, batch_size=BATCH_SIZE, shuffle=True):
    texts = data['text'].tolist()
    labels = data['hyperpartisan'].astype(int).values
    
    dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_data)
val_dataset = create_dataset(val_data, shuffle=False)
test_dataset = create_dataset(test_data, shuffle=False)

# Set up learning rate schedule and optimizer
#steps_per_epoch = len(train_data) // BATCH_SIZE
#num_train_steps = steps_per_epoch * EPOCHS
#num_warmup_steps = num_train_steps // 10

# Learning rate with polynomial decay
#
# optimizer = optimization.create_optimizer(
#     init_lr=init_lr,
#     num_train_steps=num_train_steps,
#     num_warmup_steps=num_warmup_steps,
#     optimizer_type='adamw'
# )

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
# Compile model
bert_model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# Set up callbacks
checkpoint_path = os.path.join(bert_dir, 'bert_model_checkpoint')
model_checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# Train the model
print("\nTraining BERT model...")
print(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}")
history = bert_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[model_checkpoint, early_stopping]
)

# Save the entire model
bert_model.save(os.path.join(bert_dir, 'bert_model_saved'))
print(f"Model saved to {os.path.join(bert_dir, 'bert_model_saved')}")

# Evaluate model
print("\nEvaluating BERT model on test data...")
results = bert_model.evaluate(test_dataset)
print(f"Test loss: {results[0]:.4f}")
print(f"Test accuracy: {results[1]:.4f}")
print(f"Test AUC: {results[2]:.4f}")
print(f"Test precision: {results[3]:.4f}")
print(f"Test recall: {results[4]:.4f}")
print(f"Test F1 score: {2 * results[3] * results[4] / (results[3] + results[4]):.4f}")

# Make predictions
y_true = []
y_pred = []

# Collect predictions
for texts, labels in test_dataset:
    predictions = bert_model.predict(texts)
    y_true.extend(labels.numpy())
    y_pred.extend((predictions > 0.5).astype(int).flatten())

# Classification report
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print("                   Predicted Not Hyperpartisan  Predicted Hyperpartisan")
print(f"True Not Hyperpartisan  {cm[0][0]:<27d} {cm[0][1]:<27d}")
print(f"True Hyperpartisan      {cm[1][0]:<27d} {cm[1][1]:<27d}")

# Training history summary
print("\nTraining History Summary:")
for epoch, acc, val_acc in zip(range(1, len(history.history['accuracy'])+1), 
                              history.history['accuracy'], 
                              history.history['val_accuracy']):
    print(f"Epoch {epoch}: train_accuracy={acc:.4f}, val_accuracy={val_acc:.4f}")

print("\nBERT implementation complete!")

# Function to load model and make predictions on new data (for future use)
def predict_hyperpartisan(texts, model_path=os.path.join(bert_dir, 'bert_model_saved')):
    """
    Predict hyperpartisan label for new texts
    
    Args:
        texts: List of strings containing article texts
        model_path: Path to saved BERT model
        
    Returns:
        predictions: Binary predictions (1=hyperpartisan, 0=not hyperpartisan)
        probabilities: Probability scores
    """
    # Load model
    loaded_model = tf.keras.models.load_model(model_path)
    
    # Make predictions
    probs = loaded_model.predict(texts)
    preds = (probs > 0.5).astype(int).flatten()
    
    return preds, probs