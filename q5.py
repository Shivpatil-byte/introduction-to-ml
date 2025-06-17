import numpy as np
import pandas as pd
from collections import Counter
import random

# Step 1: Create a synthetic dataset
np.random.seed(42)
texts = ["This product is amazing!"] * 50 + \
        ["I love it!"] * 25 + \
        ["Terrible experience."] * 15 + \
        ["I hate this product."] * 10
labels = ["good"] * 75 + ["bad"] * 25

data = pd.DataFrame({"Text": texts, "Label": labels})

# Step 2: Custom TF-IDF Vectorizer
def tokenize(text):
    return text.lower().replace('!', '').replace('.', '').split()

def compute_tf(text, vocab):
    tokens = tokenize(text)
    tf = Counter(tokens)
    return np.array([tf[word] / len(tokens) if word in tokens else 0 for word in vocab])

def compute_idf(corpus, vocab):
    n_docs = len(corpus)
    idf = {}
    for word in vocab:
        containing_docs = sum(1 for text in corpus if word in tokenize(text))
        idf[word] = np.log(n_docs / (1 + containing_docs))  # Smoothing
    return np.array([idf[word] for word in vocab])

def compute_tfidf(corpus):
    vocab = set(word for text in corpus for word in tokenize(text))
    vocab = list(vocab)
    tfidf = []
    idf = compute_idf(corpus, vocab)
    for text in corpus:
        tf = compute_tf(text, vocab)
        tfidf.append(tf * idf)
    return np.array(tfidf), vocab

X, vocab = compute_tfidf(data["Text"])
y = np.array(data["Label"])

# Step 3: Custom Train-Test Split
def train_test_split_manual(X, y, test_size=0.25):
    indices = list(range(len(X)))
    random.shuffle(indices)
    split = int(len(X) * (1 - test_size))
    train_indices, test_indices = indices[:split], indices[split:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.25)

# Step 4: Custom Logistic Regression
class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y = np.where(y == 'good', 1, 0)  # Convert labels to binary (1 for 'good', 0 for 'bad')

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        return np.where(predictions >= 0.5, 'good', 'bad')

# Train the model
model = LogisticRegressionCustom(learning_rate=0.1, epochs=1000)
model.train(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test)

def classification_metrics(y_true, y_pred):
    y_true = np.where(y_true == 'good', 1, 0)
    y_pred = np.where(y_pred == 'good', 1, 0)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

precision, recall, f1_score = classification_metrics(y_test, y_pred)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")

# Step 6: Predict for a single text
def predict_single_text(model, text, vocab):
    tfidf_vector = compute_tf(text, vocab) * compute_idf(data['Text'], vocab)
    tfidf_vector = tfidf_vector.reshape(1, -1)
    return model.predict(tfidf_vector)[0]

sample_text = "I loved this product!"
print(f"Prediction for '{sample_text}': {predict_single_text(model, sample_text, vocab)}")