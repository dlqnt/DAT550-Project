import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score # Added roc_auc_score
import math
import time

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
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Mixed precision enabled: {policy.name}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        target_gpu_index = 1 #GPU NUMBER use 'watch gpu-jobs' to see whats in use on gorina
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

BATCH_SIZE = 24
EPOCHS = 4 # Max epochs per fold
INIT_LR = 2e-5
PATIENCE = 2 # Early stopping patience
WEIGHT_DECAY = 0.01
SEQ_LENGTH = 512

# Cross-Validation Setup
N_SPLITS = 5 # Number of folds (K)

# Directories
data_dir = "hyperpartisan_data"
bert_dir = f"hyperpartisan_bert_models/{bert_model_name.replace('/', '_')}_CV_Ensemble_LR{INIT_LR}_BS{BATCH_SIZE}_Seq{SEQ_LENGTH}"
os.makedirs(bert_dir, exist_ok=True)
print(f"Model weights and results will be saved in: {bert_dir}")

print(f"TensorFlow version: {tf.__version__}")

# --- Data Loading and Full Preparation ---
print("\n--- Loading and Preparing Full Data ---")
articles_df = pd.read_csv(f"{data_dir}/articles_preprocessed.csv")
articles_df['text'] = articles_df['text'].fillna("")
articles_df = articles_df.reset_index(drop=True)

# --- Hold-out Test Set Split ---
print("\n--- Splitting Hold-out Test Set ---")
train_val_df, test_df = train_test_split(
    articles_df,
    test_size=0.15,
    random_state=SEED,
    stratify=articles_df['hyperpartisan']
)
print(f"Total data for Cross-Validation: {len(train_val_df)}")
print(f"Hold-out Test Set size: {len(test_df)}")
y_test_true = test_df['hyperpartisan'].astype(int).values # Store true test labels

# --- Define Functions (Dataset Creation, Model Building, Compilation) ---
def create_dataset(df, batch_size=BATCH_SIZE, shuffle=False, seed=None):
    texts = df['text'].tolist()
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

def build_bert_model(seq_length=SEQ_LENGTH):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    tokenize_layer = hub.KerasLayer(bert_preprocessor_obj.tokenize, name='tokenization')
    tokenized_inputs = tokenize_layer(text_input)
    packing_layer = hub.KerasLayer(
        bert_preprocessor_obj.bert_pack_inputs,
        arguments=dict(seq_length=seq_length),
        name='bert_packing'
    )
    encoder_inputs = packing_layer([tokenized_inputs])
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier', dtype=tf.float32)(net)
    return tf.keras.Model(inputs=text_input, outputs=net)

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

# --- Stratified K-Fold Cross-Validation ---
print(f"\n--- Starting {N_SPLITS}-Fold Stratified Cross-Validation Training ---")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

fold_val_metrics = {
    'loss': [], 'accuracy': [], 'auc': [], 'precision': [], 'recall': [], 'f1': []
}
fold_best_weight_paths = []

y = train_val_df['hyperpartisan'].values

for fold, (train_indices, val_indices) in enumerate(skf.split(train_val_df, y)):
    fold_start_time = time.time()
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

    fold_train_df = train_val_df.iloc[train_indices]
    fold_val_df = train_val_df.iloc[val_indices]
    print(f"Fold Train size: {len(fold_train_df)}, Fold Validation size: {len(fold_val_df)}")

    fold_train_dataset = create_dataset(fold_train_df, shuffle=True, seed=SEED+fold)
    fold_val_dataset = create_dataset(fold_val_df, shuffle=False)

    print("Building and Compiling Model...")
    bert_model_fold = build_bert_model(seq_length=SEQ_LENGTH)
    steps_per_epoch = math.ceil(len(fold_train_df) / BATCH_SIZE)
    num_train_steps = steps_per_epoch * EPOCHS
    bert_model_fold = compile_model_for_training( 
        bert_model_fold,
        initial_lr=INIT_LR,
        decay_steps=num_train_steps,
        weight_decay=WEIGHT_DECAY
    )

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

    print(f"Training Fold {fold+1} for up to {EPOCHS} epochs...")
    bert_model_fold.fit(
        fold_train_dataset,
        validation_data=fold_val_dataset,
        epochs=EPOCHS,
        callbacks=[model_checkpoint, early_stopping],
        verbose=1
    )

    print(f"Evaluating Fold {fold+1} on its validation set (using best weights)...")
    results = bert_model_fold.evaluate(fold_val_dataset, verbose=0)
    fold_loss, fold_acc, fold_auc, fold_prec, fold_rec = results[0], results[1], results[2], results[3], results[4]
    fold_f1 = 0.0
    if fold_prec + fold_rec > 0:
        fold_f1 = 2 * (fold_prec * fold_rec) / (fold_prec + fold_rec)
    print(f"Fold {fold+1} Validation Metrics: Loss={fold_loss:.4f}, Acc={fold_acc:.4f}, AUC={fold_auc:.4f}, Prec={fold_prec:.4f}, Rec={fold_rec:.4f}, F1={fold_f1:.4f}")

    fold_val_metrics['loss'].append(fold_loss)
    fold_val_metrics['accuracy'].append(fold_acc)
    fold_val_metrics['auc'].append(fold_auc)
    fold_val_metrics['precision'].append(fold_prec)
    fold_val_metrics['recall'].append(fold_rec)
    fold_val_metrics['f1'].append(fold_f1)

    fold_end_time = time.time()
    print(f"Fold {fold+1} completed in {fold_end_time - fold_start_time:.2f} seconds.")

 

# --- Aggregate and Report CV Results ---
print("\n--- Cross-Validation Summary (Validation Metrics) ---")
print(f"{'Metric':<12} | {'Mean':<10} | {'Std Dev':<10}")
print("-" * 35)
for metric_name in fold_val_metrics:
    mean_val = np.mean(fold_val_metrics[metric_name])
    std_val = np.std(fold_val_metrics[metric_name])
    print(f"{metric_name.capitalize():<12} | {mean_val:<10.4f} | {std_val:<10.4f}")
print("-" * 35)

# --- Ensemble Prediction on Hold-out Test Set ---
print("\n--- Generating Ensemble Predictions on Hold-out Test Set ---")
final_test_dataset = create_dataset(test_df, shuffle=False)

# Accumulator for probabilities
all_test_preds = []
inference_model = build_bert_model(seq_length=SEQ_LENGTH)

for fold, weight_path in enumerate(fold_best_weight_paths):
    print(f"Loading weights from {weight_path} and predicting (Fold {fold+1})...")
    if os.path.exists(weight_path):
        # Load the best weights for this fold into the standard model structure
        inference_model.load_weights(weight_path)

        # Predict probabilities on the test set
        fold_test_preds = []
        for texts, _ in final_test_dataset: # Iterate through batches
             preds = inference_model.predict(texts, verbose=0)
             fold_test_preds.extend(preds.flatten())
        all_test_preds.append(fold_test_preds)
    else:
        print(f"Warning: Weight file not found for fold {fold+1}: {weight_path}")

if not all_test_preds:
    print("Error: No predictions were generated from fold models. Cannot evaluate ensemble.")
else:
    # Average the probabilities across folds
    print("Averaging predictions across folds...")
    stacked_preds = np.array(all_test_preds)
    y_pred_prob_ensemble = np.mean(stacked_preds, axis=0)
    print(f"Shape of ensemble probabilities: {y_pred_prob_ensemble.shape}") # Should be (num_test_samples,)

    # Threshold averaged probabilities
    y_pred_ensemble = (y_pred_prob_ensemble > 0.5).astype(int)

    # --- Evaluate Ensemble Predictions ---
    print("\n--- Final Ensemble Performance on Hold-out Test Set ---")

    # Calculate metrics using sklearn functions for clarity
    ens_accuracy = np.mean(y_test_true == y_pred_ensemble)
    ens_precision = tf.keras.metrics.Precision()(y_test_true, y_pred_ensemble).numpy()
    ens_recall = tf.keras.metrics.Recall()(y_test_true, y_pred_ensemble).numpy()
    ens_auc = roc_auc_score(y_test_true, y_pred_prob_ensemble) # AUC needs probabilities
    ens_f1 = 0.0
    if ens_precision + ens_recall > 0:
        ens_f1 = 2 * (ens_precision * ens_recall) / (ens_precision + ens_recall)

    print(f"{'Metric':<12} | {'Score':<10}")
    print("-" * 25)
    print(f"{'Accuracy':<12} | {ens_accuracy:<10.4f}")
    print(f"{'AUC':<12} | {ens_auc:<10.4f}")
    print(f"{'Precision':<12} | {ens_precision:<10.4f}")
    print(f"{'Recall':<12} | {ens_recall:<10.4f}")
    print(f"{'F1 Score':<12} | {ens_f1:<10.4f}")
    print("-" * 25)

    print("\nDetailed Classification Report (Ensemble on Test Set):")
    print(classification_report(y_test_true, y_pred_ensemble, target_names=['Not Hyperpartisan (0)', 'Hyperpartisan (1)']))
    print("\nConfusion Matrix (Ensemble on Test Set):")
    cm_test_ens = confusion_matrix(y_test_true, y_pred_ensemble)
    print(f"{'':<25} {'Predicted Not Hyp (0)':<25} {'Predicted Hyp (1)':<25}")
    print(f"{'True Not Hyp (0)':<25} {cm_test_ens[0][0]:<25d} {cm_test_ens[0][1]:<25d}")
    print(f"{'True Hyp (1)':<25} {cm_test_ens[1][0]:<25d} {cm_test_ens[1][1]:<25d}")

print("\nCross-Validation Training and Ensemble Evaluation complete!")