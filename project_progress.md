# Hyperpartisan News Detection Project: Summary of Steps and Findings

## 1. Data Collection and Preparation
- Downloaded the SemEval hyperpartisan news detection dataset from Zenodo
- Extracted 645 news articles from XML files with their corresponding labels
- Created a structured dataset consisting of:
  - 407 non-hyperpartisan articles (63%)
  - 238 hyperpartisan articles (37%)

## 2. Text Preprocessing
- Implemented an NLP preprocessing pipeline including:
  - HTML tag removal
  - Special character filtering
  - Tokenization
  - Stopword removal
  - Lemmatization
- Preserved both original text and processed versions for analysis

## 3. Exploratory Data Analysis (EDA)

### Article Length Analysis
- **Key Finding**: Hyperpartisan articles are significantly longer
  - Hyperpartisan: 636 words on average
  - Non-hyperpartisan: 339 words on average
  - Statistically significant difference (p-value near 0)
- Vocabulary diversity is nearly identical between groups (0.729 vs. 0.733)

### Publication Patterns
- Most articles in dataset from 2016-2018
- Temporal patterns:
  - June has highest hyperpartisan ratio (0.63)
  - Thursday and Tuesday show highest hyperpartisan content (0.45 and 0.44)
  - Monday has lowest hyperpartisan ratio (0.28)

### Content Analysis
- **Distinctive hyperpartisan terms**: "trump", "american", "hillary", "clinton", "political", "liberal", "conservative", "democrat", "republican", "racist"
- **Distinctive non-hyperpartisan terms**: "police", "said", "twitter", "officer", "county", "florida", "video", "arrested"
- **Hyperpartisan distinctive phrases**: "white supremacist", "bill clinton", "ruling class", "fake news"
- **Non-hyperpartisan distinctive phrases**: "police officer", "police department", "law enforcement"

### Sentiment Analysis
- No statistically significant difference in overall sentiment (p-value = 0.7999)
- Hyperpartisan articles have:
  - Higher positive sentiment components
  - Higher negative sentiment components
- Non-hyperpartisan articles have higher neutral sentiment
- Finding: Hyperpartisan content uses more emotional language

### Correlation Analysis
- **Strongest predictors of hyperpartisan content**:
  - Article length (0.42)
  - Unique word count (0.45)
  - Positive sentiment (0.39)
  - Less neutral language (-0.46)

## 4. Feature Engineering
Created a rich feature set with 1,136 total features:
- **9 length features**: word count, sentence structure, paragraph organization
- **21 lexical features**: politically charged term counts, political figure mentions
- **6 sentiment features**: emotional content, sentiment ratio, variance
- **1,000 TF-IDF features**: vector representation of distinctive words
- **100 n-gram features**: capturing common two-word phrases

