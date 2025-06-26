import pandas as pd
import numpy as np
df=pd.read_csv("https://raw.githubusercontent.com/codebasics/py/refs/heads/master/ML/7_logistic_reg/insurance_data.csv")
df.head()
import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic Regression
class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the logistic regression model.
        Parameters:
            X: Training features (NumPy array of shape (n_samples,))
            y: Target labels (NumPy array of shape (n_samples,))
        """
        # Ensure X and y are NumPy arrays
        X = np.array(X).reshape(-1, 1)  # Reshape 1D array to 2D (n_samples, 1)
        y = np.array(y).reshape(-1, 1)  # Reshape y to 2D for calculations

        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros((n_features, 1))  # Ensure weights is a 2D array
        self.bias = 0

        # Gradient Descent
        for _ in range(self.iterations):
            # Compute the linear model
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict class labels for input features.
        Parameters:
            X: Input features (NumPy array of shape (n_samples,))
        Returns:
            Predicted binary class labels (NumPy array of shape (n_samples,))
        """
        # Ensure X is a NumPy array and reshape if needed
        X = np.array(X).reshape(-1, 1)  # Reshape 1D array to 2D (n_samples, 1)
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        return np.where(y_predicted >= 0.5, 1, 0)


if __name__ == "__main__":

    X = df.age  # 1D array of features
    y = df['bought_insurance']  # Corresponding 1D labels

    # Train the model
    model = LogisticRegression(learning_rate=0.1, iterations=1000)
    model.fit(X, y)

    # Predict for a single value
    prediction_value = 3
    prediction = model.predict(np.array([prediction_value]))
    print(f"Prediction for {prediction_value}: {prediction[0]}")
