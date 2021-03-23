import pandas as pd
import numpy as np

Filename = "1001_601128"
dp = pd.read_csv("D:/dataset/norm_data/" + Filename + ".csv")
dp_price = pd.read_csv("D:/dataset/raw_data/" + Filename + ".csv")
max_v = dp_price.iloc[:, 2].max()
min_v = dp_price.iloc[:, 2].min()
data = list()
for i in range(0, dp.shape[0]):
    temp = np.zeros(7*dp.shape[1] + 1)
    count = 0
    for k in range(0, 1):
        for item in range(dp.shape[1]):
            temp[count] = dp.values[i - k][item]
            count += 1
    temp[len(temp)-1] = dp_price.values[i][2]
    data.append(temp)
result = pd.DataFrame(data)
result.to_csv("D:/dataset/cont_data/" + Filename + "2.csv", index=False)
