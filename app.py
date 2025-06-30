import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Student Outcome Dashboard", layout="wide")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_excel("ED.xlsx")
    df["Target"] = df["Target"].map({0: "Dropout", 1: "Graduate", 2: "Enrolled"})
    return df

df = load_data()

# -------------------- SIDEBAR FILTERS --------------------
st.sidebar.title("🔎 Filter Options")
genders = st.sidebar.multiselect("Select Gender", df["Gender"].unique(), default=df["Gender"].unique())
scholarship = st.sidebar.selectbox("Scholarship Holder", ["All"] + list(df["Scholarship holder"].unique()))

filtered_df = df[df["Gender"].isin(genders)]
if scholarship != "All":
    filtered_df = filtered_df[filtered_df["Scholarship holder"] == scholarship]

# -------------------- TABS --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Overview", "📊 Demographics", "🎓 Academics", "📉 Dropouts", "📈 Correlations"])

# -------------------- TAB 1: OVERVIEW --------------------
with tab1:
    st.title("🎓 Student Outcomes Dashboard")
    st.markdown("This dashboard analyzes **student academic statuses** — Dropout, Graduate, and Enrolled — across various dimensions.")

    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(), use_container_width=True)

    st.subheader("Target Variable Distribution")
    st.write("Below is the distribution of students by their academic status.")
    st.plotly_chart(px.histogram(filtered_df, x="Target", color="Target"))

# -------------------- TAB 2: DEMOGRAPHICS --------------------
with tab2:
    st.header("📊 Demographics")

    st.subheader("Gender Distribution by Outcome")
    st.write("Distribution of student status across gender.")
    st.plotly_chart(px.histogram(filtered_df, x="Gender", color="Target", barmode="group"))

    st.subheader("Marital Status")
    st.write("Pie chart showing marital status distribution.")
    st.plotly_chart(px.pie(filtered_df, names="Marital status"))

    st.subheader("Nationality Distribution")
    st.write("Nationality vs Target status.")
    st.plotly_chart(px.histogram(filtered_df, x="Nacionality", color="Target", barmode="group"))

    st.subheader("International Students vs Target")
    st.write("Comparison of international vs local students by academic outcome.")
    st.plotly_chart(px.histogram(filtered_df, x="International", color="Target", barmode="group"))

# -------------------- TAB 3: ACADEMICS --------------------
with tab3:
    st.header("🎓 Academic Performance")

    st.subheader("Age at Enrollment")
    st.write("Age distribution across target outcomes.")
    st.plotly_chart(px.violin(filtered_df, y="Age at enrollment", color="Target", box=True))

    st.subheader("Scholarship Holder vs Target")
    st.write("Scholarship status compared to academic outcomes.")
    st.plotly_chart(px.histogram(filtered_df, x="Scholarship holder", color="Target", barmode="group"))

    st.subheader("2nd Semester Grades Distribution")
    st.write("Boxplot for grades by academic status.")
    st.plotly_chart(px.box(filtered_df, x="Target", y="Curricular units 2nd sem (grade)", color="Target"))

    st.subheader("Curricular Units Credited vs Status")
    st.write("Distribution of credited units for different student outcomes.")
    st.plotly_chart(px.box(filtered_df, x="Target", y="Curricular units 2nd sem (credited)", color="Target"))

    st.subheader("Mother's Occupation vs Target")
    st.plotly_chart(px.histogram(filtered_df, x="Mother's occupation", color="Target", barmode="group"))

    st.subheader("Father's Occupation vs Target")
    st.plotly_chart(px.histogram(filtered_df, x="Father's occupation", color="Target", barmode="group"))

# -------------------- TAB 4: DROPOUT ANALYSIS --------------------
with tab4:
    st.header("📉 Dropout Analysis")
    dropout_df = filtered_df[filtered_df["Target"] == "Dropout"]

    st.subheader("Dropouts by Gender")
    st.plotly_chart(px.pie(dropout_df, names="Gender"))

    st.subheader("Dropouts by Age")
    st.plotly_chart(px.histogram(dropout_df, x="Age at enrollment", nbins=10))

    st.subheader("Dropouts by Scholarship")
    st.plotly_chart(px.histogram(dropout_df, x="Scholarship holder", color="Gender"))

    st.subheader("Dropouts: Credited vs Grades")
    st.plotly_chart(px.scatter(dropout_df, x="Curricular units 2nd sem (credited)", y="Curricular units 2nd sem (grade)", color="Gender"))

# -------------------- TAB 5: CORRELATIONS --------------------
with tab5:
    st.header("📈 Correlation & Comparison")

    st.subheader("Correlation Heatmap")
    st.write("Correlation of numeric features to detect relationships.")
    corr = filtered_df.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    st.pyplot(fig)

    st.subheader("Feature Pairwise Scatter")
    st.write("Visualizes curricular units credited vs grades.")
    st.plotly_chart(px.scatter(filtered_df, x="Curricular units 2nd sem (credited)", y="Curricular units 2nd sem (grade)", color="Target"))

    st.subheader("Stacked Bar Chart by Gender and Status")
    gender_grouped = filtered_df.groupby(["Gender", "Target"]).size().reset_index(name="Count")
    st.plotly_chart(px.bar(gender_grouped, x="Gender", y="Count", color="Target", barmode="stack"))

    st.subheader("Histograms: Key Metrics")
    for col in ["Age at enrollment", "Curricular units 2nd sem (2)", "Curricular units 2nd sem (grade)"]:
        st.write(f"Histogram for {col}")
        st.plotly_chart(px.histogram(filtered_df, x=col, color="Target", nbins=20))
