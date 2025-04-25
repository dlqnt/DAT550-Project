import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold # Removed train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import math
import time

import tensorflow as tf
# Import the necessary pooling layers
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow_hub as hub
import tensorflow_text as text # Keep this import even if unused directly, TF Hub might need it

import getpass
import os
import logging

# Set up cache directory for TensorFlow Hub
username = getpass.getuser()
cache_dir = f'/local/home/{username}/tensorflow_hub_cache'
os.environ['TFHUB_CACHE_DIR'] = cache_dir
os.makedirs(cache_dir, exist_ok=True)

# --- GPU and Mixed Precision Setup ---
print("\n--- Setting up GPU and Mixed Precision ---")
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Mixed precision enabled: {policy.name}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        target_gpu_index = 1 # Your target index (e.g., from watch gpu-jobs)
        if target_gpu_index < len(gpus):
            visible_gpus = [gpus[target_gpu_index]]
            tf.config.set_visible_devices(visible_gpus, 'GPU')
            print(f"Attempting to use GPU: {visible_gpus[0].name}")
            tf.config.experimental.set_memory_growth(visible_gpus[0], True)
            print(f"Memory growth enabled for {visible_gpus[0].name}")
        else:
            print(f"GPU index {target_gpu_index} out of range. Available: {len(gpus)}. Check CUDA_VISIBLE_DEVICES or script.")
            if gpus: # Fallback logic
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
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- Configuration ---
print("\n--- Configuration ---")
# Model Selection (BERT Base)
bert_model_name = 'bert_en_uncased_L-12_H-768_A-12'
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
print(f"Using BERT Model: {bert_model_name}")
print(f"Encoder Handle: {tfhub_handle_encoder}")
print(f"Preprocessor Handle: {tfhub_handle_preprocess}")

# Hyperparameters (Using previously determined best)
BATCH_SIZE = 24
EPOCHS = 6 # Max epochs per fold
INIT_LR = 2e-5
PATIENCE = 2
WEIGHT_DECAY = 0.01
SEQ_LENGTH = 512
POOLING_STRATEGY = 'cls' # Best performing strategy found
print(f"Using Pooling Strategy: {POOLING_STRATEGY}")

# Cross-Validation Setup
N_SPLITS = 5

# Directories
# ---> the directory where the official split CSVs are located <---
official_data_dir = "hyperpartisan_data_official" 
# ---> Update output directory name <---
bert_dir = f"hyperpartisan_bert_models/{bert_model_name.replace('/', '_')}_OFFICIAL_SPLIT_CV_Ens_LR{INIT_LR}_BS{BATCH_SIZE}_Seq{SEQ_LENGTH}_Pool_{POOLING_STRATEGY}"
os.makedirs(bert_dir, exist_ok=True)
print(f"Model weights and results will be saved in: {bert_dir}")

print(f"TensorFlow version: {tf.__version__}")

# --- Data Loading using Official Splits ---
print("\n--- Loading Official Train/Test Data ---")
train_df_path = os.path.join(official_data_dir, "official_train_data.csv")
test_df_path = os.path.join(official_data_dir, "official_test_data.csv")

if not os.path.exists(train_df_path) or not os.path.exists(test_df_path):
    raise FileNotFoundError(f"Official train/test CSV files not found in '{official_data_dir}'. Please run the XML parsing script first.")

train_df = pd.read_csv(train_df_path)
test_df = pd.read_csv(test_df_path)

# Basic checks and cleaning - use RAW 'text' column
train_df['text'] = train_df['text'].fillna("")
test_df['text'] = test_df['text'].fillna("")
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Loaded Official Training Data: {train_df.shape}")
print(f"Loaded Official Test Data: {test_df.shape}")
print(f"Train Data - Hyperpartisan Distribution:\n{train_df['hyperpartisan'].value_counts(normalize=True)}")
print(f"Test Data - Hyperpartisan Distribution:\n{test_df['hyperpartisan'].value_counts(normalize=True)}")

# Store true test labels
y_test_true = test_df['hyperpartisan'].astype(int).values

# --- Define Functions (Dataset Creation, Model Building, Compilation) ---
def create_dataset(df, batch_size=BATCH_SIZE, shuffle=False, seed=None):
    """Creates a TF Dataset, optionally shuffled, using the 'text' column."""
    # ---> Ensure using the 'text' column containing raw text <---
    texts = df['text'].tolist()
    # -----------------------------------------------------------
    labels = df['hyperpartisan'].astype(int).values
    dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), seed=seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

print(f"Loading full preprocessor object from: {tfhub_handle_preprocess}")
bert_preprocessor_obj = hub.load(tfhub_handle_preprocess)
print("Preprocessor object loaded.")

# --- Modified build_bert_model function ---
def build_bert_model_with_pooling(
    seq_length=SEQ_LENGTH,
    pooling_strategy=POOLING_STRATEGY # Use global setting
    ):
    """Builds the BERT model with controlled sequence length and specified pooling."""
    # print(f"Building model: SeqLen={seq_length}, Pooling='{pooling_strategy}'") # Reduce verbosity inside loop
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')

    # Preprocessing
    tokenize_layer = hub.KerasLayer(bert_preprocessor_obj.tokenize, name='tokenization')
    tokenized_inputs = tokenize_layer(text_input)
    packing_layer = hub.KerasLayer(
        bert_preprocessor_obj.bert_pack_inputs,
        arguments=dict(seq_length=seq_length),
        name='bert_packing'
    )
    encoder_inputs = packing_layer([tokenized_inputs])

    # Encoder
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)

    # Pooling Strategy
    if pooling_strategy == 'avg':
        net = outputs['sequence_output']
        net = GlobalAveragePooling1D(name='global_avg_pooling')(net)
    elif pooling_strategy == 'max':
        net = outputs['sequence_output']
        net = GlobalMaxPooling1D(name='global_max_pooling')(net)
    elif pooling_strategy == 'cls':
        net = outputs['pooled_output'] # Original CLS token output
    else:
        raise ValueError(f"Unknown pooling_strategy: {pooling_strategy}. Choose 'avg', 'max', or 'cls'.")

    # Classification Head
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier', dtype=tf.float32)(net)

    return tf.keras.Model(inputs=text_input, outputs=net)

# --- Compile Function ---
def compile_model_for_training(model, initial_lr, decay_steps, weight_decay):
    """Compiles the model with AdamW and PolynomialDecay schedule."""
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        end_learning_rate=0.0,
        power=1.0
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=weight_decay
    )
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=metrics
    )
    return model

# --- Stratified K-Fold Cross-Validation (on Official Train Set) ---
print(f"\n--- Starting {N_SPLITS}-Fold Stratified Cross-Validation Training on Official Train Data ---")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

fold_val_metrics = {
    'loss': [], 'accuracy': [], 'auc': [], 'precision': [], 'recall': [], 'f1': []
}
fold_best_weight_paths = []

# ---> Use the official training data for CV splitting <---
y_train_cv = train_df['hyperpartisan'].values

for fold, (train_indices, val_indices) in enumerate(skf.split(train_df, y_train_cv)): # Split train_df
# ----------------------------------------------------
    fold_start_time = time.time()
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

    # Get fold data using iloc on train_df
    fold_train_df = train_df.iloc[train_indices]
    fold_val_df = train_df.iloc[val_indices]
    print(f"Fold Train size: {len(fold_train_df)}, Fold Validation size: {len(fold_val_df)}")

    # Create TF Datasets for the fold (using raw 'text')
    fold_train_dataset = create_dataset(fold_train_df, shuffle=True, seed=SEED+fold)
    fold_val_dataset = create_dataset(fold_val_df, shuffle=False)

    # Build and Compile Model
    print(f"Building and Compiling Model (Pooling: {POOLING_STRATEGY})...")
    bert_model_fold = build_bert_model_with_pooling( # Use the pooling function
        seq_length=SEQ_LENGTH,
        pooling_strategy=POOLING_STRATEGY
    )
    steps_per_epoch = math.ceil(len(fold_train_df) / BATCH_SIZE)
    num_train_steps = steps_per_epoch * EPOCHS
    bert_model_fold = compile_model_for_training(
        bert_model_fold,
        initial_lr=INIT_LR,
        decay_steps=num_train_steps,
        weight_decay=WEIGHT_DECAY
    )

    # Callbacks
    fold_checkpoint_path = os.path.join(bert_dir, f'fold_{fold+1}_checkpoint_best.weights.h5')
    fold_best_weight_paths.append(fold_checkpoint_path)
    model_checkpoint = ModelCheckpoint(
        filepath=fold_checkpoint_path, monitor='val_auc', mode='max',
        save_best_only=True, save_weights_only=True, verbose=0
    )
    early_stopping = EarlyStopping(
        monitor='val_auc', mode='max', patience=PATIENCE,
        restore_best_weights=True, verbose=1
    )

    # Train
    print(f"Training Fold {fold+1} for up to {EPOCHS} epochs...")
    bert_model_fold.fit(
        fold_train_dataset,
        validation_data=fold_val_dataset,
        epochs=EPOCHS,
        callbacks=[model_checkpoint, early_stopping],
        verbose=1
    )

    # Evaluate Fold
    print(f"Evaluating Fold {fold+1} on its validation set (using best weights)...")
    results = bert_model_fold.evaluate(fold_val_dataset, verbose=0)
    fold_loss, fold_acc, fold_auc, fold_prec, fold_rec = results[0], results[1], results[2], results[3], results[4]
    fold_f1 = 0.0
    if fold_prec + fold_rec > 0:
        fold_f1 = 2 * (fold_prec * fold_rec) / (fold_prec + fold_rec)
    print(f"Fold {fold+1} Validation Metrics: Loss={fold_loss:.4f}, Acc={fold_acc:.4f}, AUC={fold_auc:.4f}, Prec={fold_prec:.4f}, Rec={fold_rec:.4f}, F1={fold_f1:.4f}")

    # Store Metrics
    fold_val_metrics['loss'].append(fold_loss)
    fold_val_metrics['accuracy'].append(fold_acc)
    fold_val_metrics['auc'].append(fold_auc)
    fold_val_metrics['precision'].append(fold_prec)
    fold_val_metrics['recall'].append(fold_rec)
    fold_val_metrics['f1'].append(fold_f1)

    fold_end_time = time.time()
    print(f"Fold {fold+1} completed in {fold_end_time - fold_start_time:.2f} seconds.")


# --- Aggregate and Report CV Results ---
print("\n--- Cross-Validation Summary (Validation Metrics on Official Train Split) ---")
print(f"Pooling Strategy: {POOLING_STRATEGY}")
print(f"{'Metric':<12} | {'Mean':<10} | {'Std Dev':<10}")
print("-" * 35)
for metric_name in fold_val_metrics:
    mean_val = np.mean(fold_val_metrics[metric_name])
    std_val = np.std(fold_val_metrics[metric_name])
    print(f"{metric_name.capitalize():<12} | {mean_val:<10.4f} | {std_val:<10.4f}")
print("-" * 35)

# --- Ensemble Prediction on Official Hold-out Test Set ---
print("\n--- Generating Ensemble Predictions on Official Hold-out Test Set ---")
# ---> Use the official test_df loaded earlier <---
final_test_dataset = create_dataset(test_df, shuffle=False)
# ---------------------------------------------

all_test_preds = []
# Build Inference Model (using the specified pooling strategy)
print(f"Building inference model structure (Pooling: {POOLING_STRATEGY})...")
inference_model = build_bert_model_with_pooling(
    seq_length=SEQ_LENGTH,
    pooling_strategy=POOLING_STRATEGY
)

for fold, weight_path in enumerate(fold_best_weight_paths):
    print(f"Loading weights from {weight_path} and predicting (Fold {fold+1})...")
    if os.path.exists(weight_path):
        inference_model.load_weights(weight_path)
        fold_test_preds = []
        # Predict batch by batch on the test set
        for texts, _ in final_test_dataset:
             preds = inference_model.predict(texts, verbose=0)
             fold_test_preds.extend(preds.flatten())
        all_test_preds.append(fold_test_preds)
    else:
        print(f"Warning: Weight file not found for fold {fold+1}: {weight_path}. Skipping this fold for ensemble.")

if not all_test_preds:
    print("Error: No predictions were generated from fold models. Cannot evaluate ensemble.")
else:
    print(f"Averaging predictions across {len(all_test_preds)} folds...")
    stacked_preds = np.array(all_test_preds)
    y_pred_prob_ensemble = np.mean(stacked_preds, axis=0)
    y_pred_ensemble = (y_pred_prob_ensemble > 0.5).astype(int) # Standard threshold

    # --- Evaluate Ensemble on Official Test Set ---
    print(f"\n--- Final Ensemble Performance (Pooling: {POOLING_STRATEGY}) on Official Hold-out Test Set ---")
    # ---> Use y_test_true defined from the official test_df <---
    ens_accuracy = np.mean(y_test_true == y_pred_ensemble)
    # Use TF metrics for consistency or sklearn directly
    ens_precision_metric = tf.keras.metrics.Precision()
    ens_precision_metric.update_state(y_test_true, y_pred_ensemble)
    ens_precision = ens_precision_metric.result().numpy()

    ens_recall_metric = tf.keras.metrics.Recall()
    ens_recall_metric.update_state(y_test_true, y_pred_ensemble)
    ens_recall = ens_recall_metric.result().numpy()

    ens_auc = roc_auc_score(y_test_true, y_pred_prob_ensemble) # AUC needs probabilities
    ens_f1 = 0.0
    if ens_precision + ens_recall > 0:
        ens_f1 = 2 * (ens_precision * ens_recall) / (ens_precision + ens_recall)
    # --------------------------------------------------------

    print(f"{'Metric':<12} | {'Score':<10}")
    print("-" * 25)
    print(f"{'Accuracy':<12} | {ens_accuracy:<10.4f}")
    print(f"{'AUC':<12} | {ens_auc:<10.4f}")
    print(f"{'Precision':<12} | {ens_precision:<10.4f}") # Hyperpartisan class if using default threshold=0.5
    print(f"{'Recall':<12} | {ens_recall:<10.4f}")    # Hyperpartisan class if using default threshold=0.5
    print(f"{'F1 Score':<12} | {ens_f1:<10.4f}")       # Hyperpartisan class if using default threshold=0.5
    print("-" * 25)

    print("\nDetailed Classification Report (Ensemble on Official Test Set):")
    print(classification_report(y_test_true, y_pred_ensemble, target_names=['Not Hyperpartisan (0)', 'Hyperpartisan (1)']))
    print("\nConfusion Matrix (Ensemble on Official Test Set):")
    cm_test_ens = confusion_matrix(y_test_true, y_pred_ensemble)
    print(f"{'':<25} {'Predicted Not Hyp (0)':<25} {'Predicted Hyp (1)':<25}")
    print(f"{'True Not Hyp (0)':<25} {cm_test_ens[0][0]:<25d} {cm_test_ens[0][1]:<25d}")
    print(f"{'True Hyp (1)':<25} {cm_test_ens[1][0]:<25d} {cm_test_ens[1][1]:<25d}")

print("\nCross-Validation Training and Ensemble Evaluation on Official Splits complete!")