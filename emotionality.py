from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt


def extract_emotionality(text: str) -> pd.Series:
    """
    Analyze text polarity and subjectivity via TextBlob.
    Returns a Series [polarity, subjectivity].
    """
    blob = TextBlob(text)
    return pd.Series([blob.sentiment.polarity, blob.sentiment.subjectivity],
                     index=["polarity", "subjectivity"])


def get_emotionality_report(df: pd.DataFrame):
    """
    Train a logistic regression on polarity and subjectivity,
    return the classification report as DataFrame along with model and test data.
    """
    # Drop incomplete rows and extract features
    df = df.dropna(subset=["text", "hyperpartisan"])
    df[["polarity", "subjectivity"]] = df["text"].apply(extract_emotionality)

    # Train/test split
    X = df[["polarity", "subjectivity"]]
    y = df["hyperpartisan"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # Fit model
    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)

    # Predict and report
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = (
        pd.DataFrame(report)
          .transpose()
          .reset_index()
          .rename(columns={"index": "Class"})
    )

    return report_df, model, X_test, y_test


def load_bert_report(path: str) -> pd.DataFrame:
    """
    Load BERT results CSV and normalize column names.
    """
    df_bert = pd.read_csv(path)
    return (
        df_bert
          .rename(columns={
              "Precision": "precision_bert",
              "Recall":    "recall_bert",
              "F1-Score":  "f1_bert"
          })
    )


def merge_reports(emotion_df: pd.DataFrame, bert_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map class names to a common scheme, then merge emotion vs. BERT metrics.
    """
    name_map = {
        "0":            "Not Hyperpartisan (0)",
        "1":            "Hyperpartisan (1)",
        "accuracy":     "Accuracy",
        "macro avg":    "Macro Avg",
        "weighted avg": "Weighted Avg"
    }
    emotion_df["Class"] = emotion_df["Class"].replace(name_map)
    emotion_df = emotion_df.rename(columns={
        "precision": "precision_emotionality",
        "recall":    "recall_emotionality",
        "f1-score":  "f1_emotionality"
    })
    return pd.merge(emotion_df, bert_df, on="Class")


def plot_comparison(df: pd.DataFrame):
    """
    Plot a side-by-side bar chart for a given metric (precision|recall|f1).
    """
    metrics = ["precision", "recall", "f1"]
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    for ax, metric in zip(axes, metrics):
        ax.bar([i - 0.25*1 for i in range(len(df))], [0.5] * len(df), 0.25, label="Random", alpha=1)
        ax.bar([i - 0.25*0 for i in range(len(df))], df[f"{metric}_emotionality"], 0.25, label="Emotionality", alpha=1)
        ax.bar([i + 0.25*1 for i in range(len(df))], df[f"{metric}_bert"], 0.25, label="BERT", alpha=1)
        # If a model was completely random:
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f"{metric.capitalize()} Comparison", fontsize=14)
        ax.legend(loc="upper left")
    axes[-1].set_xticks(range(len(df)))
    axes[-1].set_xticklabels(df["Class"], rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_roc(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Plot ROC curve and compute AUC for the emotionality model.
    """
    # Probability estimates for the positive class
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, label=f"Emotionality ROC (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Emotionality Model", fontsize=14)
    ax.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    data_path = "hyperpartisan_data/articles_byarticle.csv"
    bert_path = "hyperpartisan_data/results/final_ensemble.csv"

    # Load data
    df = pd.read_csv(data_path)

    # Emotionality report + model
    emotion_df, model, X_test, y_test = get_emotionality_report(df)

    # BERT report
    bert_df = load_bert_report(bert_path)

    # Merge for comparison
    comp_df = merge_reports(emotion_df, bert_df)

    # Show merged metrics
    print(comp_df[[
        "Class",
        "precision_emotionality", "precision_bert",
        "recall_emotionality",    "recall_bert",
        "f1_emotionality",        "f1_bert"
    ]])

    plot_comparison(comp_df)
    # The emotionality model is better than random, and does not require much resources.
    # The BERT model is better than random, but requires more resources.


    # Plot ROC curve
    plot_roc(model, X_test, y_test) 

if __name__ == "__main__":
    main()