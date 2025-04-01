# Hyperpartisan News Detection - Data Exploration Summary

# Key Findings
## Dataset Composition

677 partisan sites/publishers in total
Strong political imbalance: 499 right-leaning (74%) vs 178 left-leaning (26%) sites
81 Macedonian sites (suspected fake news sources), with 79 of them being right-leaning

## Facebook Engagement

452 Facebook pages associated with these partisan sites
490 matches when merging (some sites have multiple pages)
Facebook page "about" sections average only 14 words, but contain useful political signals (e.g., "conservative" appears 52 times)

## Data Structure & Quality

Some sites have multiple Facebook pages (10 sites with exactly 2 pages)
Missing values present (118 missing Facebook IDs)
Good metadata available through WHOIS data (registration dates, contact info)

## Challenges for the Project

Missing Article Content: The BuzzFeed dataset provides publisher metadata but lacks the actual news article text needed for hyperpartisan classification.
Class Imbalance: The significant imbalance between right and left-leaning sources (3:1 ratio) will require careful handling in the model.
Data Quality: Type conversion issues and missing values need to be addressed in preprocessing.

## Recommended Next Steps
1. Acquire Article Text Data
The BuzzFeed dataset doesn't include actual article content. Options:

Download the SemEval-2019 Task 4 datasets mentioned in your project description
Use the by-article dataset (1,273 manually labeled articles)
Use the by-publisher dataset (754,000 articles with distant supervision)

2. Data Preprocessing

Clean HTML content from articles
Handle missing values
Create appropriate train/validation/test splits

3. Feature Engineering
Based on successful approaches in the SemEval paper:

Text-based features: TF-IDF, word embeddings (Word2Vec, GloVe, ELMo, BERT)
Stylometric features: Readability measures, sentence length, punctuation patterns
Sentiment and emotion: Analyze emotional language (successful in SemEval)
Link patterns: Check if articles reference known partisan sites (effective in top SemEval entries)
Metadata: Publication date, domain registration information

4. Model Implementation
Following the SemEval findings:

Baseline models: Naive Bayes, Decision Trees (required by project)
Advanced models: CNN with embeddings (the winning approach used ELMo + CNN)
Hybrid approaches: Combine linguistic features with deep learning models

5. Evaluation Strategy

Use metrics required by the project: precision, recall, F1, ROC curve
Implement cross-validation
Perform error analysis to identify challenging cases

## Immediate Action Items

Download SemEval datasets: This is critical as we need article text content
Set up preprocessing pipeline: Create code to clean and prepare text data
Implement baseline text features: Start with TF-IDF representation
Develop evaluation framework: Create splits and evaluation metric