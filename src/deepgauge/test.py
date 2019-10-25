import csv
import pandas as pd
values = pd.read_csv("/home/pass-la-1/PycharmProjects/mydeepconcolic/result/fashion_mnist/74_1-0_50_new.csv", header=None).to_numpy()
values = (values * 250).astype(int)

with open("/home/pass-la-1/PycharmProjects/mydeepconcolic/result/fashion_mnist/74_1-0_50_new222.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(values)