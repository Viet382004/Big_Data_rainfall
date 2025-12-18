import pandas as pd
df = pd.read_csv('rainfall_clean.csv')

def descriptive(df):
    df_num = df.select_dtypes(include = 'number')
    df_min = df_num.min()
    df_max = df_num.max()
    df_median = df_num.median()
    df_mean = df_num.mean()
    df_mode = df_num.mode()
    df_q1 = df_num.quantile(0.25)
    df_q2 = df_num.quantile(0.5)
    df_q3 = df_num.quantile(0.75)
    df_iqr = df_q3 -df_q1
    df_var = df_num.var()
    df_stdev = df_num.std()

    data = {
        "Min":[i for i in df_min],
        "Max":[i for i in df_max],
        "Mean":[i for i in df_mean],
        "Median":[i for i in df_median],
        "Mode":[i for i in df_mode],
        "q1":[i for i in df_q1],
        "q2":[i for i in df_q2],
        "q3":[i for i in df_q3],
        "iqr":[i for i in df_iqr],
        "variance": [i for i in df_var],
        "std_dev": [i for i in df_stdev],

    }
    df_data = pd.DataFrame(data)
    df_data.index = df_num.keys()
    df_complete = df_data.transpose()
    print("Bảng lược đồ tóm lược dữ liệu của cột dữ liệu dạng số:")
    print(df_complete.to_string())
    return df_complete
descriptive(df)