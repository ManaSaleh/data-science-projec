import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load the dataset
@st.cache_data
def load_data():
    file_path = "../Data/clening_data.csv"
    data = pd.read_csv(file_path)
    # Drop unnecessary columns if they exist
    data = data.drop(columns=["Unnamed: 0", "Student ID", "Program ID"], errors="ignore")
    return data

# Set up the app layout
st.set_page_config(
    page_title="EDA with Plotly",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App Title
st.title("📊 Data Exploration App")
st.markdown("### Analyze your dataset with modern visualizations and interactivity")

# Load the data
df = load_data()

# Dataset preview
st.sidebar.header("🗂️ Dataset Overview")
st.sidebar.write(f"Number of rows: **{df.shape[0]}**")
st.sidebar.write(f"Number of columns: **{df.shape[1]}**")
if st.sidebar.checkbox("Show sample data"):
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.sample(10))

# Sidebar - Column Selection
st.sidebar.header("📈 Column Analysis")
selected_column = st.sidebar.selectbox("Select a column to explore:", df.columns)

# Display column details
if selected_column:
    st.subheader(f"Exploring Column: {selected_column}")
    st.write(f"**Data Type:** {df[selected_column].dtype}")
    st.write(f"**Unique Values:** {df[selected_column].nunique()}")
    st.write(f"**Sample Values:** {df[selected_column].unique()[:10]}")

    # Numeric or categorical analysis
    if pd.api.types.is_numeric_dtype(df[selected_column]):
        st.write(f"**Summary Statistics:**")
        st.write(df[selected_column].describe())
        
        # Distribution plot
        st.write("**Distribution**")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[selected_column], nbinsx=30, name="Distribution"))
        fig.update_layout(
            title=f"Distribution of {selected_column}",
            xaxis_title=selected_column,
            yaxis_title="Count",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("**Value Counts**")
        value_counts = df[selected_column].value_counts().reset_index()
        value_counts.columns = [selected_column, "count"]  # Rename columns for clarity
        st.dataframe(value_counts)

        # Bar chart for value counts
        fig = px.bar(
            value_counts,
            x=selected_column,
            y="count",
            labels={selected_column: selected_column, "count": "Count"},
            title=f"Value Counts for {selected_column}",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

# Sidebar - Relationship Analysis
st.sidebar.header("🔗 Relationship Analysis")
col1 = st.sidebar.selectbox("Select Column 1:", df.columns, key="col1")
col2 = st.sidebar.selectbox("Select Column 2:", df.columns, key="col2")

# Display relationship analysis
if col1 and col2:
    st.subheader(f"Relationship Between {col1} and {col2}")
    if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[col1], y=df[col2], mode="markers", name="Scatter"))
        fig.update_layout(
            title=f"Scatter Plot: {col1} vs {col2}",
            xaxis_title=col1,
            yaxis_title=col2,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.histogram(
            df,
            x=col1,
            color=col2,
            barmode="group",
            title=f"Distribution of {col1} by {col2}",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

# Enhanced Analysis with Y Column
st.sidebar.header("🎯 Target Analysis: Y")
if "Y" in df.columns:
    # Summary Statistics for Y
    st.subheader("Target Column: Y (Program Completion)")
    st.write(
        """
        **Definition:**  
        - `1`: Did not complete the program.  
        - `0`: Successfully completed the program.
        """
    )
    y_counts = df["Y"].value_counts()
    y_percentages = df["Y"].value_counts(normalize=True) * 100

    st.write("**Distribution of Y:**")
    st.write(f"**Total Entries:** {len(df)}")
    st.write(f"- **0 (Completed):** {y_counts[0]} ({y_percentages[0]:.2f}%)")
    st.write(f"- **1 (Not Completed):** {y_counts[1]} ({y_percentages[1]:.2f}%)")

    # Pie chart of Y distribution
    fig_pie = px.pie(
        names=["Completed", "Not Completed"],
        values=y_counts,
        title="Program Completion Distribution",
        hole=0.4,
        template="plotly_white",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Select a column to analyze with Y
    st.sidebar.subheader("Analyze Relation Between a Column and Y")
    target_analysis_col = st.sidebar.selectbox(
        "Select a column to compare with Y:", 
        [col for col in df.columns if col != "Y"],
        key="target_col"
    )

    if target_analysis_col:
        st.subheader(f"Relationship Between `{target_analysis_col}` and Y")

        # Numerical column analysis with Y
        if pd.api.types.is_numeric_dtype(df[target_analysis_col]):
            st.write(f"**Summary Statistics of `{target_analysis_col}` by Y:**")
            st.write(df.groupby("Y")[target_analysis_col].describe())

            # Box plot
            fig_box = px.box(
                df,
                x="Y",
                y=target_analysis_col,
                title=f"Box Plot: `{target_analysis_col}` vs Y",
                labels={"Y": "Target (Y)", target_analysis_col: target_analysis_col},
                template="plotly_white",
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # Categorical column analysis with Y
        else:
            st.write(f"**Value Counts of `{target_analysis_col}` by Y:**")
            grouped_counts = df.groupby([target_analysis_col, "Y"]).size().reset_index(name="count")
            st.dataframe(grouped_counts)

            # Stacked bar chart
            fig_bar = px.bar(
                grouped_counts,
                x=target_analysis_col,
                y="count",
                color="Y",
                barmode="stack",
                title=f"Stacked Bar Chart: `{target_analysis_col}` by Y",
                labels={"count": "Count", target_analysis_col: target_analysis_col, "Y": "Completion Status"},
                template="plotly_white",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.sidebar.warning("Column 'Y' is not in the dataset.")

# Footer
st.markdown("---")
st.markdown("**Created by [Your Name]**")
st.markdown("[GitHub Repository](https://github.com/your-repo)")

