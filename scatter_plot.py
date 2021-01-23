import pandas as pd
import numpy as np
from matplotlib import pyplot

def scatter(df):
    fig = pyplot.figure(figsize=(15,5))
    i = 0
    for column in df:
        pyplot.scatter(df[column], np.full(len(df),i), label=column)
        i += 1
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.legend()
    pyplot.show()

def main():
    try:
        df = pd.read_csv("resources/dataset_train.csv", index_col=0)
    except:
        exit("Error: Something went wrong with the dataset")
    df = df.drop(columns=["Hogwarts House","First Name","Last Name","Birthday","Best Hand"])
    scatter(df)
    df = df.drop(columns=["Arithmancy"])
    scatter(df)
    df = df.drop(columns=["Muggle Studies", "Astronomy"])
    scatter(df)
    df = df.drop(columns=["Flying", "Transfiguration", "Ancient Runes"])
    scatter(df)
    df = df.drop(columns=["Charms"])
    scatter(df)
    df = df.drop(columns=["Potions", "Care of Magical Creatures"])
    scatter(df)
    df = df.drop(columns=["Defense Against the Dark Arts", "Divination"])
    scatter(df)

    

if __name__ == "__main__":
    main()
