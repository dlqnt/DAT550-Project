from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd

# The purpose of this script is to have a basis of comparison for the BERT model.

# The path to the data
path = "hyperpartisan_data/"
df = pd.read_csv(path + "articles_byarticle.csv")

# Dropping dataframes that does not have the text or hyperpartisan label
df = df.dropna(subset=['text', 'hyperpartisan'])

def extract_emotionality(text):
    '''
    This function uses TextBlob to extract the polarity and subjectivity of the text.
    In other words, if the text uses positive or negative words, rather than neutral words,
    TextBlob will return a value between -1 and 1, where -1 is very negative, 0 is neutral,

    Parameters:
    text (str): The text to analyze.

    Returns:
    pd.Series: A series containing the polarity and subjectivity of the text.
    '''
    # Using TextBlob to analyze the text
    blob = TextBlob(text)
    # Get the sentiment of the text blob
    sentiment = blob.sentiment

    # Extract polarity and subjectivity
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity

    # Return as a pandas Series
    return pd.Series([polarity, subjectivity])

# Apply the function to the text column
df[['polarity', 'subjectivity']] = df['text'].apply(extract_emotionality)

# Convert hyperpartisan labels to binary
X = df[['polarity', 'subjectivity']]
y = df['hyperpartisan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

# Train a logistic regression model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
