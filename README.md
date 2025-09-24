# IMDb Movie Review Sentiment Classification

The project applies a sentiment classification pipeline based on Python's scikit-learn library to label IMDb movie reviews as positive or negative. The objective is to train a Logistic Regression model based on the IMDb Movie Review Dataset and test its accuracy.

## Approach

### Data
The data employed is the IMDb Movie Review Dataset, provided as `IMDB Dataset.csv`. It has two columns:
- **review**: The movie review text.
- **sentiment**: The sentiment tag, either "positive" or "negative" (translated to 1 and 0, respectively).

The data is drawn from a local file (`IMDB Dataset.csv`) in this repository.

### Steps in Preprocessing
Raw text data is preprocessed to prepare it for a machine learning model:
1. **Text Cleaning**: The punctuation is removed and text is changed to lowercase using regular expressions to minimize noise.
2. **Text Vectorization**: Applies `TfidfVectorizer` to convert reviews to numerical features. This process:
- Creates a vocabulary up to the top 10,000 words (`max_features=10000`).
- Adds unigrams and bigrams (`ngram_range=(1, 2)`) to include phrases.
- Drops common English stop words to concentrate on useful words.
3. **Label Encoding**: Converts positive and negative sentiments into 1 and 0 for binary classification.
4. **Train-Test Split**: Splits the data into 80% training and 20% test sets for assessment.

### Model Used
The algorithm used is **Logistic Regression**, a linear classifier optimized for binary problems. The salient features are:
- Hyperparameter tuning with `GridSearchCV` to determine the optimal `C` value (strength of regularization) among [0.01, 0.1, 1, 10, 100].
- Utilizes the `liblinear` solver for a speed boost on small datasets.
- Includes `class_weight='balanced'` to account for possible class imbalance.

### Results
The model has a performance of around 89.02% on the test set, with possible enhancements investigated (e.g., text lemmatization, n-gram extension). Cross-validation is employed to maximize robustness.

### How to Run
1. Install Python and necessary libraries:
2. Place the 'IMDB Dataser.csv' file in the projec directory
3. Run the script:
4. The script outputs the steps and final accuracy.

### Repository
This project is hosted at: [https://github.com/srisaidhakshini/IMDb_movie_review](https://github.com/srisaidhakshini/IMDb_movie_review).

### Resources
- Logistic Regression: [https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- Text Vectorization: [https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)