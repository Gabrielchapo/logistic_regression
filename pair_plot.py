import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    try:
        df = pd.read_csv("resources/dataset_train.csv", index_col=0)
    except:
        exit("Error: Something went wrong with the dataset")
    df = df.drop(columns=["First Name","Last Name","Birthday","Best Hand"])
    g = sns.pairplot(df, hue="Hogwarts House")
    plt.show()
 

if __name__ == "__main__":
    main()
