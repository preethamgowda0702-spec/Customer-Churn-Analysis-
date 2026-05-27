import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("customer_churn_dataset-testing-master.csv")

print(df.head())

print(df.info())

df["Churn"].value_counts().plot(kind="bar")

plt.show()
