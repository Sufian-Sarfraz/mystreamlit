import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def load_data():
    return pd.read_csv("Student_Assessment_Preprocessed.csv")

df = load_data()


numeric_df = df.copy()
for column in numeric_df.select_dtypes(include=['object']).columns:
    numeric_df[column] = numeric_df[column].astype('category').cat.codes


st.title("Instructor Dashboard for E-Learning Systems")

st.sidebar.header("Dashboard Filters")
age_filter = st.sidebar.multiselect("Select Age Band", df['Age_band'].unique(), default=df['Age_band'].unique())
edu_filter = st.sidebar.multiselect("Select Education Level", df['Highest_education'].unique(), default=df['Highest_education'].unique())


filtered_data = df[(df['Age_band'].isin(age_filter)) & (df['Highest_education'].isin(edu_filter))]
filtered_numeric_data = numeric_df[(df['Age_band'].isin(age_filter)) & (df['Highest_education'].isin(edu_filter))]


st.subheader("Dataset Overview")
st.write("Summary Statistics:")
st.write(filtered_data.describe())
st.write("Filtered Dataset Preview:")
st.dataframe(filtered_data)


st.subheader("Student Result Distribution")
result_counts = filtered_data['Student_final_result'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(result_counts, labels=result_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
plt.title("Student Final Results Distribution")
st.pyplot(plt)


st.subheader("Feature Correlation Heatmap")
plt.figure(figsize=(10, 6))
sns.heatmap(filtered_numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
st.pyplot(plt)


st.subheader("Classifier Accuracy Comparison")
models = ['Logistic Regression', 'SVM', 'Random Forest']
accuracies = [0.85, 0.88, 0.92]  # Example values; replace with actual results
colors = ['blue', 'green', 'red']
plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=colors)
plt.ylim(0.8, 1.0)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
st.pyplot(plt)


st.subheader("Predict Student Performance")
student_input = st.text_input("Enter Student Data (Comma-separated values):")
if st.button("Predict"):
   
    prediction = "Pass"  
    st.success(f"Predicted Result: {prediction}")


st.subheader("Additional Visualizations")

plt.figure(figsize=(8, 5))
sns.histplot(filtered_data['Assessment_score'], kde=True, bins=20, color='skyblue')
plt.title("Distribution of Assessment Scores")
plt.xlabel("Assessment Score")
plt.ylabel("Frequency")
st.pyplot(plt)
