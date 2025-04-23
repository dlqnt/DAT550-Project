import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import pandas as pd

# Load the Hyperpartisan News Dataset
df = pd.read_csv('hyperpartisan_data/articles_preprocessed.csv')

# Drop rows missing data to avoid NaNs
df = df.dropna(subset=['processed_text', 'hyperpartisan'])

# Convert boolean labels to integers
df['label'] = df['hyperpartisan'].astype(int)

# Split into train/validation/test sets
texts = df['processed_text'].astype(str).tolist()
labels = df['label'].values
# First split: train+val vs test
X_temp, X_test, y_temp, y_test_true = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
# Second split: train vs validation from the temp set
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# Tokenize and pad sequences
vocab_size = 10000
max_length = 200
oov_token = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(X_train)

def prepare(sequences):
    seq = tokenizer.texts_to_sequences(sequences)
    return pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')

X_train_pad = prepare(X_train)
X_val_pad = prepare(X_val)
X_test_pad = prepare(X_test)

# Build the LSTM model
embedding_dim = 128  # Embedding dimension
lstm_units = 64      # LSTM units

def build_model():
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = build_model()
model.summary()

# Train the model
epochs = 10
batch_size = 64
history = model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_val_pad, y_val),
    epochs=epochs,
    batch_size=batch_size
)

# Save the trained model and tokenizer
df_save = 'models'
model.save(f'{df_save}/hyperpartisan_lstm.h5')
import pickle
with open(f'{df_save}/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Training complete. Model and tokenizer saved.")

# Evaluate on hold-out test set
# Generate probability predictions
y_pred_prob_ensemble = model.predict(X_test_pad).flatten()
# Convert probabilities to binary predictions
y_pred_ensemble = np.where(y_pred_prob_ensemble >= 0.5, 1, 0)

print(f"\n--- Final Ensemble Performance ---")

# Compute metrics
ess_accuracy = np.mean(y_test_true == y_pred_ensemble)
ess_precision = tf.keras.metrics.Precision()(y_test_true, y_pred_ensemble).numpy()
ess_recall = tf.keras.metrics.Recall()(y_test_true, y_pred_ensemble).numpy()
ess_auc = roc_auc_score(y_test_true, y_pred_prob_ensemble)
ess_f1 = 0.0
if ess_precision + ess_recall > 0:
    ens_f1 = 2 * (ess_precision * ess_recall) / (ess_precision + ess_recall)

print(f"{'Metric':<12} | {'Score':<10}")
print("-" * 25)
print(f"{'Accuracy':<12} | {ess_accuracy:<10.4f}")
print(f"{'AUC':<12} | {ess_auc:<10.4f}")
print(f"{'Precision':<12} | {ess_precision:<10.4f}")
print(f"{'Recall':<12} | {ess_recall:<10.4f}")
print(f"{'F1 Score':<12} | {ens_f1:<10.4f}")
print("-" * 25)

print("\nDetailed Classification Report (Ensemble on Test Set):")
print(classification_report(y_test_true, y_pred_ensemble, target_names=['Not Hyperpartisan (0)', 'Hyperpartisan (1)']))

print("\nConfusion Matrix (Ensemble on Test Set):")
cm_test_ens = confusion_matrix(y_test_true, y_pred_ensemble)
print(f"{'':<25} {'Predicted Not Hyp (0)':<25} {'Predicted Hyp (1)':<25}")
print(f"{'True Not Hyp (0)':<25} {cm_test_ens[0][0]:<25d} {cm_test_ens[0][1]:<25d}")
print(f"{'True Hyp (1)':<25} {cm_test_ens[1][0]:<25d} {cm_test_ens[1][1]:<25d}")
