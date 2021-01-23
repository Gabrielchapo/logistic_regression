import pandas as pd
from matplotlib import pyplot

try:
    df = pd.read_csv("resources/dataset_train.csv", index_col=0)
except:
    exit("Error: Something went wrong with the dataset")

fig = pyplot.figure(figsize=(15,5))

df = df.drop(columns=["First Name","Last Name","Birthday","Best Hand"]).fillna(df.mean())

for i in range(1,14):

    ax = fig.add_subplot(2,7,i)
    tmp = df.groupby(["Hogwarts House"])[df.columns[i]]
    list_to_plot = [data for name, data in tmp]
    list_label = [name for name, data in tmp]
    ax.hist(list_to_plot, label=list_label, histtype='bar')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.title.set_text(df.columns[i])

pyplot.show()