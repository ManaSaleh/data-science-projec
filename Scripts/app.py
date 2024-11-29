import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load the dataset
@st.cache_data
def load_data():
    file_path = "../Data/clening_data.csv"
    data = pd.read_csv(file_path)
    # Drop 'Unnamed: 0' and 'Student ID' columns if they exist
    data = data.drop(columns=["Unnamed: 0", "Student ID" , "Program ID"], errors="ignore")
    return data

# App Title
st.title("Data Exploration App with Plotly")

# Load the data
df = load_data()
st.write("Dataset Preview:")
st.dataframe(df.sample(10))

# Column Selection
st.sidebar.header("Column Analysis")
selected_column = st.sidebar.selectbox("Select a column to explore:", df.columns)

# Display column details
if selected_column:
    st.subheader(f"Exploring Column: {selected_column}")
    st.write(f"Data Type: {df[selected_column].dtype}")
    st.write(f"Unique Values: {df[selected_column].nunique()}")
    st.write(f"Sample Values: {df[selected_column].unique()[:10]}")

    # Display Distribution for numerical or categorical columns
    if pd.api.types.is_numeric_dtype(df[selected_column]):
        st.write(f"Summary Statistics:")
        st.write(df[selected_column].describe())
        st.write("Distribution:")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[selected_column], nbinsx=30, name="Distribution"))
        fig.update_layout(title=f"Distribution of {selected_column}", xaxis_title=selected_column, yaxis_title="Count")
        st.plotly_chart(fig)
    else:
        st.write("Value Counts:")
        st.write(df[selected_column].value_counts())
        fig = px.bar(
            df[selected_column].value_counts().reset_index(),
            x="index",
            y=selected_column,
            labels={"index": selected_column, selected_column: "Count"},
            title=f"Value Counts for {selected_column}",
        )
        st.plotly_chart(fig)

# Relationship Analysis
st.sidebar.header("Column Relationship Analysis")
col1 = st.sidebar.selectbox("Select Column 1:", df.columns)
col2 = st.sidebar.selectbox("Select Column 2:", df.columns)

if col1 and col2:
    st.subheader(f"Relationship Between {col1} and {col2}")
    if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[col1], y=df[col2], mode="markers", name="Scatter"))
        fig.update_layout(
            title=f"Scatter Plot: {col1} vs {col2}",
            xaxis_title=col1,
            yaxis_title=col2,
        )
        st.plotly_chart(fig)
    else:
        fig = px.histogram(
            df,
            x=col1,
            color=col2,
            barmode="group",
            title=f"Distribution of {col1} by {col2}",
        )
        st.plotly_chart(fig)

# Analysis with Y Column
st.sidebar.header("Relation with Target Column Y")
target_analysis_col = st.sidebar.selectbox("Select a column to compare with Y:", [col for col in df.columns if col != "Y"])

if target_analysis_col:
    st.subheader(f"Relationship Between {target_analysis_col} and Y")
    if pd.api.types.is_numeric_dtype(df[target_analysis_col]):
        fig = px.box(
            df,
            x="Y",
            y=target_analysis_col,
            title=f"Box Plot: {target_analysis_col} vs Y",
            labels={"Y": "Target (Y)", target_analysis_col: f"{target_analysis_col}"},
        )
        st.plotly_chart(fig)
    else:
        fig = px.histogram(
            df,
            x=target_analysis_col,
            color="Y",
            barmode="group",
            title=f"Distribution of {target_analysis_col} by Y",
        )
        st.plotly_chart(fig)
