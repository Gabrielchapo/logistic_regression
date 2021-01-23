import pandas as pd
import numpy as np
import sys
from LogisticRegression import LogisticRegression

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

    model.cross_validation(X, Y, 0.10, 3000, nb_folds=10)
    model.save()


