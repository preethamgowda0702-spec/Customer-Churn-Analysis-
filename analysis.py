import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("customer_churn_dataset-testing-master.csv")

# First 5 rows
print("First 5 Rows")
print(df.head())

# Churn count
print("\nChurn Count")
print(df["Churn"].value_counts())

# Churn percentage
churn_percentage = (df["Churn"].value_counts(normalize=True) * 100)

print("\nChurn Percentage")
print(churn_percentage)

# Graph 1 - Churn Count
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Count")
plt.savefig("churn_count.png")
plt.clf()

# Graph 2 - Gender vs Churn
sns.countplot(x="Gender", hue="Churn", data=df)
plt.title("Gender vs Churn")
plt.savefig("gender_churn.png")
plt.clf()

# Graph 3 - Total Spend vs Churn
sns.boxplot(x="Churn", y="Total Spend", data=df)
plt.title("Total Spend vs Churn")
plt.savefig("total_spend.png")
plt.clf()

# Customer insights
print("\nCustomer Insights")
print("1. Customers with higher spending tend to churn more.")
print("2. Gender-based churn patterns can be analyzed.")
print("3. Churn analysis helps companies improve retention.")

# Conclusion
print("\nConclusion")
print("Customer churn analysis helps businesses identify customers likely to leave and improve retention strategies.")

print("\nGraphs saved successfully.")