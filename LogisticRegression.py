import numpy as np

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def error(Y_pred, Y_real):
    logged = - np.log(Y_pred[np.arange(Y_real.shape[0]), Y_real.argmax(axis=1)])
    return np.sum(logged) / Y_real.shape[0]

class LogisticRegression:

    def fit(self, X, Y, lr, epochs, verbose=0):

        self.sigma = [np.amax(x) - np.amin(x) if np.amax(x) - np.amin(x) != 0 else 1 for x in zip(*X)]
        self.mean = [sum(x) / len(X) for x in zip(*X)]
        X = (X - self.mean) / self.sigma
        X = np.insert(X, X.shape[1], 1, axis=1)
        self.weights = np.random.randn(X.shape[1], Y.shape[1])
        for i in range(epochs):
            predicted = np.dot(X, self.weights)
            predicted = sigmoid(predicted)
            diff = predicted - Y
            gradient_vector = np.dot(X.T, diff)
            self.weights -= (lr / X.shape[0]) * gradient_vector
            if verbose == 1:
                print("Epoch:", i + 1, "/", epochs, "=== Loss:", error(predicted, Y))
    
    def predict(self, X):
        X = (X - self.mean) / self.sigma
        return np.dot(X, self.weights[:-1]) + self.weights[-1]

    def save(self):
        dict = {"weights":self.weights, "mean":self.mean, "sigma":self.sigma}
        np.save("resources/weights.npy", dict)

    def load(self):
        dict = np.load("resources/weights.npy", allow_pickle='TRUE').item()
        self.sigma = dict["sigma"]
        self.mean = dict["mean"]
        self.weights = dict["weights"]

