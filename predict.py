import pandas as pd
import numpy as np
import sys
from LogisticRegression import LogisticRegression

if __name__ == "__main__":
    try:
        df = pd.read_csv(sys.argv[1], index_col=0)
    except:
        exit("Error: Something went wrong with the dataset")

    l = ['Gryffindor', 'Ravenclaw', 'Slytherin', 'Hufflepuff']
    df = df.drop(columns=["Hogwarts House","First Name","Last Name","Birthday"]).fillna(0)
    X = pd.get_dummies(df).to_numpy()

    model = LogisticRegression()
    model.load()
    tmp = model.predict(X)
    for i,x in enumerate(tmp):
        print(l[np.argmax(x)])
