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
    
    def evaluate(self, X_test, Y_test):
        predictions = self.predict(X_test)
        predictions = [np.argmax(x) for x in predictions]
        Y_test = [np.argmax(x) for x in Y_test]
        count = 0
        for i in range(len(Y_test)):
            if Y_test[i] == predictions[i]:
                count += 1
        return count / len(Y_test)
    
    def cross_validation(self, X, Y, lr, epochs, nb_folds=5):
        folds_size = len(X) // nb_folds
        all_accuracies = []
        for index in range(nb_folds):
            X_train = np.array([x for i,x in enumerate(X) if i <= index * folds_size or i > (index+1) * folds_size])
            Y_train = np.array([y for i,y in enumerate(Y) if i <= index * folds_size or i > (index+1) * folds_size])
            X_test = np.array([x for i,x in enumerate(X) if i > index * folds_size and i <= (index+1) * folds_size])
            Y_test = np.array([y for i,y in enumerate(Y) if i > index * folds_size and i <= (index+1) * folds_size])
            self.fit(X_train,Y_train, lr, epochs, verbose=1)
            all_accuracies.append(self.evaluate(X_test, Y_test))
        print("all accuracies from cross validation: ", all_accuracies)
        print("Mean accuracy: ", sum(all_accuracies) / len(all_accuracies))
    
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

