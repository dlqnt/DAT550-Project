import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import pandas as pd
import pickle

# Load and clean
train_df = pd.read_csv('hyperpartisan_data_official/official_train_data.csv')
test_df  = pd.read_csv('hyperpartisan_data_official/official_test_data.csv')
train_df = train_df.dropna(subset=['text', 'hyperpartisan'])
test_df  = test_df.dropna(subset=['text', 'hyperpartisan'])
train_df['label'] = train_df['hyperpartisan'].astype(int)
test_df['label']  = test_df['hyperpartisan'].astype(int)

# Split TRAIN into train and validation
texts    = train_df['text'].astype(str).tolist()
labels   = train_df['label'].values
texts_test  = test_df['text'].astype(str).tolist()
labels_test = test_df['label'].values

X_train_texts, X_val_texts, y_train, y_val = train_test_split(
    texts, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Tokenize on the TRAINING TEXTS only
vocab_size  = 10_000
max_length  = 200
oov_token   = '<OOV>'
tokenizer   = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(X_train_texts)

def prepare(text_list):
    seq = tokenizer.texts_to_sequences(text_list)
    return pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')

X_train_pad = prepare(X_train_texts)
X_val_pad   = prepare(X_val_texts)
X_test_pad  = prepare(texts_test)

# Build the LSTM
embedding_dim = 512 
lstm_units    = 64 * 5

def build_model():
    m = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        Bidirectional(LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m

model = build_model()
model.summary()

# Train
class_weight = {0:1.0, 1:1.0}
history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=8, batch_size=64,
    class_weight=class_weight
)

# Save
# model.save('models/hyperpartisan_lstm.keras')
# with open('models/tokenizer.pkl', 'wb') as f:
#     pickle.dump(tokenizer, f)

print("Training complete. Model and tokenizer saved.")

# Evaluate on the TEST set
y_pred_prob = model.predict(X_test_pad).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)

acc  = (labels_test == y_pred).mean()
auc  = roc_auc_score(labels_test, y_pred_prob)
prec = tf.keras.metrics.Precision()(labels_test, y_pred).numpy()
rec  = tf.keras.metrics.Recall()(labels_test, y_pred).numpy()
f1   = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

print(f"\n{'Metric':<12} | {'Score':<10}")
print("-" * 25)
print(f"{'Accuracy':<12} | {acc:<10.4f}")
print(f"{'AUC':<12} | {auc:<10.4f}")
print(f"{'Precision':<12} | {prec:<10.4f}")
print(f"{'Recall':<12} | {rec:<10.4f}")
print(f"{'F1 Score':<12} | {f1:<10.4f}")
print("-" * 25)

print("\nClassification Report:")
print(classification_report(labels_test, y_pred,
                            target_names=['Not Hyperpartisan', 'Hyperpartisan']))

print("\nConfusion Matrix:")
cm = confusion_matrix(labels_test, y_pred)
print(f"{'':<25} {'Predicted Not Hyp (0)':<25} {'Predicted Hyp (1)':<25}")
print(f"{'True Not Hyp (0)':<25} {cm[0][0]:<25d} {cm[0][1]:<25d}")
print(f"{'True Hyp (1)':<25} {cm[1][0]:<25d} {cm[1][1]:<25d}")
