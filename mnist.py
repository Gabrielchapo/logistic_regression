import pandas as pd
import numpy as np
import sys
from LogisticRegression import LogisticRegression

if __name__ == "__main__":
    try:
        df = pd.read_csv(sys.argv[1])
    except:
        exit("Error: Something went wrong with the dataset")
    tmp = df["label"].tolist()
    Y = np.zeros((len(tmp), 10))
    for i,x in enumerate(tmp):
        Y[i][x] = 1
    X = df.drop(columns=["label"]).to_numpy()

    Y_train, Y_test = Y[:2500], Y[2500:]
    X_train, X_test = X[:2500], X[2500:]
    model = LogisticRegression()

    model.fit(X_train, Y_train, 0.20, 1000, verbose=1)
    model.save()
    tmp = model.predict(X_test)
    count = 0
    for i,x in enumerate(tmp):
        print(np.argmax(x), np.argmax(Y_test[i]), end=" ")
        if np.argmax(x) == np.argmax(Y_test[i]):
            print("SUCCESS")
            count += 1
        else:
            print("FAILURE")

    print("Accuracy: ", count / len(tmp))

