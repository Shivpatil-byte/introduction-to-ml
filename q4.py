import numpy as np
import pandas as pd
from collections import Counter
import random

# Step 1: Create a synthetic dataset
np.random.seed(42)
reviews = ["This movie was amazing!"] * 25 + \
          ["I absolutely loved it!"] * 25 + \
          ["The film was terrible."] * 25 + \
          ["I hated this movie."] * 25
sentiments = ["positive"] * 50 + ["negative"] * 50

# Combine into a DataFrame
data = pd.DataFrame({
    "Review": reviews,
    "Sentiment": sentiments
})

# Step 2: Tokenize and create a vocabulary
def tokenize(review):
    return review.lower().replace('!', '').replace('.', '').split()

# Create a vocabulary
all_words = [word for review in data['Review'] for word in tokenize(review)]
vocab = list(set(all_words))  # Unique words

# Step 3: Create feature vectors
def vectorize(review, vocab):
    tokens = tokenize(review)
    return np.array([tokens.count(word) for word in vocab])

# Generate feature vectors for all reviews
X = np.array([vectorize(review, vocab) for review in data['Review']])
y = np.array(data['Sentiment'])

# Step 4: Split the dataset manually
indices = list(range(len(X)))
random.shuffle(indices)
split = int(0.8 * len(indices))
train_indices = indices[:split]
test_indices = indices[split:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Step 5: Train the Naive Bayes classifier
class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.word_probs = {}

    def train(self, X, y):
        classes = np.unique(y)
        total_samples = len(y)

        # Compute class probabilities
        self.class_probs = {cls: np.sum(y == cls) / total_samples for cls in classes}

        # Compute word probabilities for each class
        word_counts = {cls: np.sum(X[y == cls], axis=0) for cls in classes}
        total_words = {cls: np.sum(word_counts[cls]) for cls in classes}

        self.word_probs = {
            cls: (word_counts[cls] + 1) / (total_words[cls] + len(vocab))  # Laplace smoothing
            for cls in classes
        }

    def predict(self, X):
        results = []
        for sample in X:
            class_scores = {}
            for cls in self.class_probs:
                class_scores[cls] = np.log(self.class_probs[cls]) + \
                                    np.sum(sample * np.log(self.word_probs[cls]))
            results.append(max(class_scores, key=class_scores.get))
        return np.array(results)

# Train the classifier
nb_classifier = NaiveBayesClassifier()
nb_classifier.train(X_train, y_train)

# Step 6: Evaluate the model
y_pred = nb_classifier.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy on the test set: {accuracy:.2f}")

# Step 7: Function to predict sentiment for a single review
def predict_review_sentiment(nb_classifier, vocab, review):
   
    review_vector = vectorize(review, vocab)
    return nb_classifier.predict([review_vector])[0]

# Test the function
sample_review = "I really enjoyed this movie!"
predicted_sentiment = predict_review_sentiment(nb_classifier, vocab, sample_review)
print(f"The sentiment for the review '{sample_review}' is predicted to be: {predicted_sentiment}")
