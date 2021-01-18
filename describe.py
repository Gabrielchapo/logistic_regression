import math
import pandas as pd
import sys

def describe(df):
    
    df = df._get_numeric_data()
    legend = {'legend':['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']}
    df_described = pd.DataFrame(data=legend)

    real_size = df.shape[0]

    for column in df:

        mean = 0
        count = 0
        min_value, max_value = float('inf'), float('-inf')
        first_quartile = None
        second_quartile = None
        third_quartile = None
        sorted_column = df[column].sort_values()
        for i, row in enumerate(sorted_column):
            if pd.notna(row):
                if i == int(real_size * 0.25) and first_quartile == None:
                    first_quartile = row
                if i == int(real_size * 0.5) and second_quartile == None:
                    second_quartile = row
                if i == int(real_size * 0.75) and third_quartile == None:
                    third_quartile = row
                if row < min_value:
                    min_value = row
                if row > max_value:
                    max_value = row
                mean += row
                count += 1
        if mean > 0:
            mean /= count
        tmp = 0
        for row in df[column]:
            if pd.notna(row):
                tmp += (row - mean) ** 2
        std = math.sqrt(tmp / (count - 1))

        df_described[column] = [count, mean, std, min_value, first_quartile, second_quartile, third_quartile, max_value]

    pd.set_option('display.max_columns', None)
    print(df_described.to_string(index=False))

if __name__ == "__main__":

    try:
        df = pd.read_csv(sys.argv[1], index_col=0)
    except:
        exit("Error: Something went wrong with the dataset")
    describe(df)