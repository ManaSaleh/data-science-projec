import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load the dataset
@st.cache_data
def load_data():
    file_path = "Data/cleaning_data.csv"
    data = pd.read_csv(file_path)
    # Drop unnecessary columns if they exist
    data = data.drop(columns=["Unnamed: 0", "Student ID", "Program ID"], errors="ignore")
    return data

# Set up the app layout
st.set_page_config(
    page_title="EDA with Plotly",
    page_icon="📊",
    layout="wide",
)

# App Title
st.title("📊 Data Exploration App")
st.markdown(
    """
    ### Discover insights and relationships within your dataset using interactive visualizations.
    Use the tabs below to explore the dataset in detail, analyze relationships between columns, and generate powerful charts.
    """
)

# Load the data
df = load_data()

# Dataset Overview
st.sidebar.header("🗂️ Dataset Overview")
st.sidebar.write(f"**Number of Rows:** {df.shape[0]}")
st.sidebar.write(f"**Number of Columns:** {df.shape[1]}")
if st.sidebar.checkbox("Show Dataset Summary"):
    st.subheader("Dataset Summary")
    st.write(df.describe(include="all"))

if st.sidebar.checkbox("Show Sample Data"):
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.sample(10))

# Dataset Description in Sidebar
st.sidebar.header("📘 Dataset Description")
st.sidebar.markdown("""
This dataset delves into students' registration information and program participation at Tuwaiq Academy. Each entry includes details like personal info, program specifics, and academic achievements. Analyze attendance, technology types, and employment statuses to uncover patterns for optimizing educational experiences.
""")

# Tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Single Column Analysis",
    "🔗 Multi-Column Relationships",
    "📊 Advanced Visualizations",
    "🎯 Y Analysis"
])

# Tab 1: Single Column Analysis
with tab1:
    st.subheader("🔍 Single Column Exploration")
    selected_column = st.selectbox("Select a column to explore:", df.columns)

    if selected_column:
        st.write(f"**Column Type:** {df[selected_column].dtype}")
        st.write(f"**Unique Values:** {df[selected_column].nunique()}")

        # Numeric Column
        if pd.api.types.is_numeric_dtype(df[selected_column]):
            st.write("**Summary Statistics:**")
            st.write(df[selected_column].describe())

            bins = st.slider("Adjust Number of Bins", min_value=5, max_value=100, value=30)
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[selected_column], nbinsx=bins, name="Distribution"))
            fig.update_layout(
                title=f"Distribution of {selected_column}",
                xaxis_title=selected_column,
                yaxis_title="Count",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Categorical Column
            value_counts = df[selected_column].value_counts().reset_index()
            value_counts.columns = [selected_column, "Count"]
            st.dataframe(value_counts)

            fig = px.bar(
                value_counts,
                x=selected_column,
                y="Count",
                title=f"Value Counts for {selected_column}",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

# Tab 2: Multi-Column Relationships
with tab2:
    st.subheader("🔗 Multi-Column Relationships")
    col1 = st.selectbox("Select Column 1:", df.columns, key="col1")
    col2 = st.selectbox("Select Column 2:", df.columns, key="col2")
    col3 = st.selectbox("Optional: Select Column 3:", ["None"] + list(df.columns), key="col3")

    if col1 and col2:
        st.write(f"Exploring relationships between `{col1}`, `{col2}`" + (f", and `{col3}`" if col3 != "None" else ""))

        if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
            fig = px.scatter(
                df,
                x=col1,
                y=col2,
                color=col3 if col3 != "None" else None,
                title=f"Scatter Plot: {col1} vs {col2}" + (f" by {col3}" if col3 != "None" else ""),
                template="plotly_white",
                marginal_x="histogram",
                marginal_y="histogram",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            grouped = df.groupby([col1, col2] + ([col3] if col3 != "None" else [])).size().reset_index(name="Count")
            fig = px.bar(
                grouped,
                x=col1,
                y="Count",
                color=col2,
                facet_col=col3 if col3 != "None" else None,
                title=f"Bar Chart: {col1} by {col2}" + (f" and {col3}" if col3 != "None" else ""),
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

# Tab 3: Advanced Visualizations
with tab3:
    st.subheader("📊 Advanced Multi-Column Visualizations")

    # Numeric aggregation
    col = st.selectbox("Select a column to aggregate:", df.select_dtypes(include=["number"]).columns)
    agg_func = st.selectbox("Select an aggregation function:", ["mean", "sum", "count", "max", "min"])
    group_by_cols = st.multiselect("Group by columns:", [c for c in df.columns if c != col])

    if col and group_by_cols:
        try:
            agg_df = df.groupby(group_by_cols)[col].agg(agg_func).reset_index()
            st.write("### Aggregated Data")
            st.dataframe(agg_df)

            vis_type = st.selectbox("Choose a visualization type:", ["Bar Chart", "Line Chart", "Heatmap"])
            if vis_type == "Bar Chart":
                fig = px.bar(
                    agg_df,
                    x=group_by_cols[0],
                    y=col,
                    color=group_by_cols[1] if len(group_by_cols) > 1 else None,
                    title=f"{agg_func.capitalize()} of {col} grouped by {' and '.join(group_by_cols)}",
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)
            elif vis_type == "Line Chart":
                fig = px.line(
                    agg_df,
                    x=group_by_cols[0],
                    y=col,
                    color=group_by_cols[1] if len(group_by_cols) > 1 else None,
                    markers=True,
                    title=f"{agg_func.capitalize()} of {col} grouped by {' and '.join(group_by_cols)}",
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)
            elif vis_type == "Heatmap" and len(group_by_cols) > 1:
                pivot_table = agg_df.pivot(index=group_by_cols[0], columns=group_by_cols[1], values=col)
                fig = go.Figure(
                    data=go.Heatmap(
                        z=pivot_table.values,
                        x=pivot_table.columns,
                        y=pivot_table.index,
                        colorscale="Viridis",
                    )
                )
                fig.update_layout(
                    title=f"Heatmap of {agg_func.capitalize()} {col}",
                    xaxis_title=group_by_cols[1],
                    yaxis_title=group_by_cols[0],
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)
            elif vis_type == "Heatmap":
                st.warning("Heatmap requires at least two grouping columns.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please select at least one grouping column.")

# Tab 4: Y Analysis
with tab4:
    st.subheader("🎯 Target Analysis: Y")
    st.markdown("""
    - `1`: Did not complete the program.
    - `0`: Successfully completed the program.
    """)

    # Distribution of Y
    y_counts = df["Y"].value_counts()
    y_percentages = df["Y"].value_counts(normalize=True) * 100

    st.write("**Distribution of Y:**")
    st.write(f"**Total Entries:** {len(df)}")
    st.write(f"- **0 (Completed):** {y_counts[0]} ({y_percentages[0]:.2f}%)")
    st.write(f"- **1 (Not Completed):** {y_counts[1]} ({y_percentages[1]:.2f}%)")

    # Pie chart for Y distribution
    fig_pie = px.pie(
        names=["Completed", "Not Completed"],
        values=y_counts,
        title="Program Completion Distribution (Y)",
        hole=0.4,
        template="plotly_white",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Analyze relationship between Y and other columns
    st.sidebar.header("Analyze Relationship with Y")
    analysis_col = st.sidebar.selectbox("Select a column to compare with Y:", [col for col in df.columns if col != "Y"])

    if analysis_col:
        st.subheader(f"Relationship Between `{analysis_col}` and Y")

        # Numerical column analysis
        if pd.api.types.is_numeric_dtype(df[analysis_col]):
            st.write(f"**Summary Statistics of `{analysis_col}` by Y:**")
            st.write(df.groupby("Y")[analysis_col].describe())

            # Box plot
            fig_box = px.box(
                df,
                x="Y",
                y=analysis_col,
                title=f"Box Plot: `{analysis_col}` vs Y",
                labels={"Y": "Program Completion (Y)", analysis_col: analysis_col},
                template="plotly_white",
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # Categorical column analysis
        else:
            st.write(f"**Value Counts of `{analysis_col}` by Y:**")
            grouped_counts = df.groupby([analysis_col, "Y"]).size().reset_index(name="Count")
            st.dataframe(grouped_counts)

            # Stacked bar chart
            fig_bar = px.bar(
                grouped_counts,
                x=analysis_col,
                y="Count",
                color="Y",
                barmode="stack",
                title=f"Stacked Bar Chart: `{analysis_col}` by Y",
                labels={"Count": "Count", analysis_col: analysis_col, "Y": "Completion Status (Y)"},
                template="plotly_white",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
