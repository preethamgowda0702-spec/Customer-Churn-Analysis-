import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("📊 Customer Churn Analysis and Prediction Dashboard")

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("customer_churn_dataset-testing-master.csv")

# -----------------------------
# DATASET PREVIEW
# -----------------------------
st.header("1. Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# CHURN COUNT
# -----------------------------
st.header("2. Churn Count")

churn_counts = df["Churn"].value_counts()
st.write(churn_counts)

fig1, ax1 = plt.subplots()
sns.countplot(x="Churn", data=df, ax=ax1)
ax1.set_title("Customer Churn Distribution")
st.pyplot(fig1)

# -----------------------------
# CHURN PERCENTAGE
# -----------------------------
st.header("3. Churn Percentage")

churn_percentage = df["Churn"].value_counts(normalize=True) * 100
st.write(churn_percentage)

# -----------------------------
# GENDER VS CHURN
# -----------------------------
st.header("4. Gender vs Churn")

fig2, ax2 = plt.subplots()
sns.countplot(x="Gender", hue="Churn", data=df, ax=ax2)
st.pyplot(fig2)

# -----------------------------
# TOTAL SPEND VS CHURN
# -----------------------------
st.header("5. Total Spend vs Churn")

fig3, ax3 = plt.subplots()
sns.boxplot(x="Churn", y="Total Spend", data=df, ax=ax3)
st.pyplot(fig3)

# -----------------------------
# CUSTOMER INSIGHTS
# -----------------------------
st.header("6. Customer Insights")

st.write("Average Age:", round(df["Age"].mean(), 2))
st.write("Average Total Spend:", round(df["Total Spend"].mean(), 2))
st.write("Average Tenure:", round(df["Tenure"].mean(), 2))

# -----------------------------
# MACHINE LEARNING SECTION
# -----------------------------
st.header("7. Customer Churn Prediction (ML)")

try:
    ml_df = df.copy()

    # Encode all text columns
    for col in ml_df.select_dtypes(include=["object"]).columns:
        ml_df[col] = LabelEncoder().fit_transform(
            ml_df[col].astype(str)
        )

    # Remove CustomerID if present
    if "CustomerID" in ml_df.columns:
        ml_df = ml_df.drop("CustomerID", axis=1)

    X = ml_df.drop("Churn", axis=1)
    y = ml_df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression(max_iter=5000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    st.success(
        f"Model Accuracy: {accuracy * 100:.2f}%"
    )

    # -----------------------------
    # ACTUAL VS PREDICTED
    # -----------------------------
    st.header("8. Actual vs Predicted")

    result_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    })

    st.dataframe(result_df.head(20))

    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------
    st.header("9. Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig4, ax4 = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax4
    )

    ax4.set_xlabel("Predicted")
    ax4.set_ylabel("Actual")

    st.pyplot(fig4)

    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------
    st.header("10. Feature Importance")

    importance = abs(model.coef_[0])

    feature_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )

    st.dataframe(feature_df)

    fig5, ax5 = plt.subplots()
    sns.barplot(
        data=feature_df,
        x="Importance",
        y="Feature",
        ax=ax5
    )

    st.pyplot(fig5)

except Exception as e:
    st.error(f"Machine Learning Error: {e}")

# -----------------------------
# CONCLUSION
# -----------------------------
st.header("11. Conclusion")

st.write("""
Customer churn analysis helps businesses identify
customers who are likely to leave a service.
Using visualization and machine learning,
organizations can improve customer retention
and make better business decisions.
""")