import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())

print(df.info())

df["Churn"].value_counts().plot(kind="bar")

plt.show()
