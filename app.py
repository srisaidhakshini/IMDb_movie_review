import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load the dataset
print("Step 1: Loading the IMDb Movie Review Dataset...")
data = pd.read_csv(r"C:\Users\sunda\OneDrive\ドキュメント\project_2\IMDB Dataset.csv", encoding='utf-8')
X_raw = data['review'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
y = data['sentiment'].map({'positive': 1, 'negative': 0})

# 2. Preprocess: Use TfidfVectorizer with n-grams
print("\nStep 2: Vectorizing the text data using TfidfVectorizer...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
X = vectorizer.fit_transform(X_raw)

# 3. Split the data
print("\nStep 3: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train with hyperparameter tuning
print("\nStep 4: Training the Logistic Regression model with GridSearchCV...")
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
model = GridSearchCV(LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'), param_grid, cv=5)
model.fit(X_train, y_train)
print(f"Best C: {model.best_params_['C']}")

# 5. Evaluate
print("\nStep 5: Making predictions and evaluating performance...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")