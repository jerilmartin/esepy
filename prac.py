import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, os
from dicttoxml import dicttoxml

st.set_page_config(page_title="Generic Data Dashboard", layout="wide")


st.sidebar.title("Load Data")
file = st.sidebar.file_uploader("Upload CSV", type="csv")

if file: 
    df = pd.read_csv(file)
    st.success("CSV Loaded Successfully")
else:
    st.warning("Please upload a CSV file to proceed.")

if 'df' in locals():
    st.title(" Data Management & Visualization Dashboard")


    st.subheader("Data Cleaning")

    if st.checkbox("Remove Duplicate Rows"):
        before = len(df)
        df.drop_duplicates(inplace=True)
        after = len(df)
        st.success(f"Removed {before - after} duplicate rows")

    if st.checkbox("Fill Missing Values"):
        df = df.apply(lambda col: col.fillna(col.mean()) if col.dtype != 'object' else col.fillna("Unknown"))
        st.success("Missing values filled")


    st.subheader("Dataset Overview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.json({"Columns": list(df.columns), 
             "Missing values": df.isnull().sum().to_dict()})


    st.subheader("Save / Download Dataset")
    os.makedirs("exports", exist_ok=True)

    csv_file = "exports/data.csv"
    if st.button("Save as CSV"):
        df.to_csv(csv_file, index=False)
        st.success("Saved CSV")
    st.download_button("Download CSV", df.to_csv(index=False), 
                       file_name="data.csv", mime="text/csv")

    if st.button("Append to CSV"):
        df.to_csv(csv_file, mode="a", index=False, header=False)
        st.success("Appended CSV")

    pkl_file = "exports/data.pkl"
    if st.button("Save as Pickle"):
        with open(pkl_file, "wb") as f: pickle.dump(df, f)
        st.success("Saved Binary (Pickle)")
    st.download_button("Download Pickle", pickle.dumps(df), 
                       file_name="data.pkl", mime="application/octet-stream")

    xml_file = "exports/data.xml"
    if st.button("Save as XML"):
        xml_data = dicttoxml(df.to_dict("records"), custom_root="records", attr_type=False)
        with open(xml_file, "wb") as f: f.write(xml_data)
        st.success("Saved XML")
    st.download_button("Download XML", dicttoxml(df.to_dict("records"), custom_root="records", attr_type=False), 
                       file_name="data.xml", mime="application/xml")

 
    st.subheader("Analysis & Visualization")
    st.write("Summary Stats:")
    st.dataframe(df.describe(include="all"))


    st.subheader("Data Manipulations")
    sort_column = st.selectbox("Sort by column:", df.columns.tolist())
    if st.button("Sort Data"):
        st.dataframe(df.sort_values(by=sort_column, ascending=False))

    st.write("Slice Rows by index")
    start_idx = st.number_input("Start index:", min_value=0, max_value=len(df)-1, value=0)
    end_idx = st.number_input("End index:", min_value=0, max_value=len(df), value=5)
    if st.button("Show Slice"):
        st.dataframe(df.iloc[start_idx:end_idx])


    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        selected_col = st.selectbox("Filter by column:", cat_cols)
        selected_val = st.selectbox("Choose value:", df[selected_col].unique())
        if st.button("Filter Data"):
            st.dataframe(df.loc[df[selected_col]==selected_val])


    st.subheader("Visualization")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if num_cols:
        plot_option = st.selectbox("Choose Plot", ["Histogram", "Scatter", "Bar"])

        if plot_option == "Histogram":
            col = st.selectbox("Column for Histogram:", num_cols)
            fig, ax = plt.subplots()
            ax.hist(df[col].dropna(), bins=10, color="skyblue", edgecolor="black")
            ax.set_title(f"{col} Distribution")
            st.pyplot(fig)

        elif plot_option == "Scatter" and len(num_cols) >= 2:
            x_col = st.selectbox("X-axis:", num_cols)
            y_col = st.selectbox("Y-axis:", num_cols)
            fig, ax = plt.subplots()
            ax.scatter(df[x_col], df[y_col], alpha=0.6, color="green")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{x_col} vs {y_col}")
            st.pyplot(fig)

        elif plot_option == "Bar":
            col = st.selectbox("Column for Bar Plot:", num_cols)
            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind="bar", ax=ax, color="orange")
            ax.set_title(f"{col} Frequency")
            st.pyplot(fig)

    else:
        st.info("No numeric columns available for plotting.")


    st.markdown("---")
    st.subheader("Extra Queries (General Purpose)")


    col_choice = st.selectbox("Choose column for Min/Max:", df.columns)
    if st.button("Show Min/Max"):
        if np.issubdtype(df[col_choice].dtype, np.number):
            st.write({"Min": df[col_choice].min(), "Max": df[col_choice].max()})
        else:
            st.warning("Selected column is not numeric.")


    col_stats = st.selectbox("Choose column for Statistics:", df.columns)
    if st.button("Show Mean/Median/Std"):
        if np.issubdtype(df[col_stats].dtype, np.number):
            st.write({
                "Mean": df[col_stats].mean(),
                "Median": df[col_stats].median(),
                "Std Dev": df[col_stats].std()
            })
        else:
            st.warning("Selected column is not numeric.")


    filter_col = st.selectbox("Choose column for filtering:", df.columns)
    condition = st.selectbox("Condition:", [">", "<", "="])
    filter_val = st.text_input("Enter value:")
    if st.button("Apply Filter"):
        try:
            if np.issubdtype(df[filter_col].dtype, np.number):
                val = float(filter_val)
            else:
                val = filter_val
            if condition == ">":
                filtered = df[df[filter_col] > val]
            elif condition == "<":
                filtered = df[df[filter_col] < val]
            else:
                filtered = df[df[filter_col] == val]
            st.dataframe(filtered)
        except Exception as e:
            st.error(f"Error in filtering: {e}")
