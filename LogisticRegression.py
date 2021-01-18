import numpy as np
import pandas as pd
import sys

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

if __name__ == "__main__":
    try:
        df = pd.read_csv(sys.argv[1], index_col=0)
    except:
        exit("Error: Something went wrong with the dataset")

    houses = df["Hogwarts House"].tolist()
    l = ['Gryffindor', 'Ravenclaw', 'Slytherin', 'Hufflepuff']
    Y = np.array([np.array([1,0,0,0]) if x == 'Gryffindor' else np.array([0,1,0,0]) if x == 'Ravenclaw' else np.array([0,0,1,0]) if x == 'Slytherin' else np.array([0,0,0,1]) for x in houses])
    df = df.drop(columns=["Hogwarts House","First Name","Last Name","Birthday"]).fillna(0)
    X = pd.get_dummies(df).to_numpy()

    model = LogisticRegression()

    model.fit(X, Y, 0.05, 3000, verbose=1)
    tmp = model.predict(X)
    for i,x in enumerate(tmp):
        print(houses[i], np.argmax(x), end=" ")
        if houses[i] == l[np.argmax(x)]:
            print("CORRECT")
        else:
            print("FALSE")

