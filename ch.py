import streamlit as st
import pandas as pd
import numpy as np
import pickle, io
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


def xml_to_df(xml_bytes):
    """Convert simple XML into DataFrame"""
    root = ET.fromstring(xml_bytes)
    rows = []
    for item in root:
        rows.append({child.tag: child.text for child in item})
    return pd.DataFrame(rows)

def df_to_xml(df):
    """Convert DataFrame into XML string"""
    root = ET.Element("data")
    for _, row in df.iterrows():
        item = ET.SubElement(root, "row")
        for col, val in row.items():
            child = ET.SubElement(item, col)
            child.text = str(val)
    return ET.tostring(root, encoding="utf-8")


st.set_page_config(page_title="📊 Easy Data Dashboard", layout="wide")
st.title("📊 Easy Data Dashboard")


st.sidebar.header("Upload Data")
file = st.sidebar.file_uploader("Upload CSV/TXT/TSV/PKL/XML", 
                                type=["csv","txt","tsv","pkl","xml"])

df = pd.DataFrame()
if file:
    if file.name.endswith(("csv", "txt", "tsv")):
        sep = "\t" if file.name.endswith(("tsv", "txt")) else ","
        df = pd.read_csv(file, sep=sep)
    elif file.name.endswith(".pkl"):
        df = pickle.load(file)
    elif file.name.endswith(".xml"):
        df = xml_to_df(file.read())

if st.sidebar.button("🔄 Reset Data"):
    st.experimental_rerun()

with st.expander("📌 Dataset Preview", expanded=True):
    st.dataframe(df)
    if not df.empty:
        st.write(f"**Shape:** {df.shape}")
        st.write("**Summary:**")
        st.write(df.describe(include="all"))


with st.expander("➕ Append New Row"):
    if not df.empty:
        with st.form("append_row", clear_on_submit=True):
            new_row = {col: st.text_input(f"{col}") for col in df.columns}
            if st.form_submit_button("Add Row"):
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                st.success("Row added successfully!")
                st.dataframe(df.tail())
    else:
        st.info("Upload data first to add rows.")


with st.expander("🩹 Handle Missing Values"):
    if not df.empty:
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            col = st.selectbox("Column with NaN", missing_cols)
            action = st.radio("Action", ["Fill Mean","Fill Median","Fill Mode","Drop Rows","Drop Column"])
            if st.button("Apply Missing Value Fix"):
                if action == "Fill Mean": df[col].fillna(df[col].mean(), inplace=True)
                elif action == "Fill Median": df[col].fillna(df[col].median(), inplace=True)
                elif action == "Fill Mode": df[col].fillna(df[col].mode()[0], inplace=True)
                elif action == "Drop Rows": df.dropna(subset=[col], inplace=True)
                elif action == "Drop Column": df.drop(columns=[col], inplace=True)
                st.success(f"{action} applied!")
        else:
            st.info("No missing values found.")

with st.expander("🔧 Transformations & Filters"):
    if not df.empty:
        # Convert type
        c = st.selectbox("Column to convert", df.columns)
        t = st.selectbox("Convert to", ["int","float","str","datetime"])
        if st.button("Convert Column Type"):
            df[c] = pd.to_datetime(df[c], errors="coerce") if t=="datetime" else df[c].astype(t)
            st.success(f"Converted {c} to {t}")

        # Sort
        sort_cols = st.multiselect("Sort by", df.columns)
        if st.button("Sort Data") and sort_cols:
            df = df.sort_values(by=sort_cols)

        # Group
        if st.checkbox("Group By"):
            g = st.selectbox("Group column", df.columns)
            num_cols = df.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                a = st.selectbox("Aggregate column", num_cols)
                f = st.selectbox("Function", ["mean","sum","count","min","max"])
                st.write(getattr(df.groupby(g)[a], f)())

        # Slice
        start, end = st.slider("Row Slice", 0, len(df), (0, min(len(df),10)))
        st.write(df.iloc[start:end])

        # Filter
        expr = st.text_input("Filter expression (e.g., age > 22)")
        if expr:
            try: st.dataframe(df.query(expr))
            except: st.error("Invalid expression")


with st.expander("📈 Visualization"):
    if not df.empty:
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            x = st.selectbox("X-axis", df.columns)
            y = st.selectbox("Y-axis", [c for c in num_cols if c != x], index=0)
            chart = st.radio("Chart Type", ["Histogram","Bar","Line","Scatter"])
            if st.button("Plot Chart"):
                fig, ax = plt.subplots()
                if chart == "Histogram": ax.hist(df[x].dropna(), bins=20)
                elif chart == "Bar": ax.bar(df[x].astype(str), df[y])
                elif chart == "Line": ax.plot(df[x], df[y])
                else: ax.scatter(df[x], df[y])
                st.pyplot(fig)


with st.expander("📊 NumPy / Pandas Ops"):
    if not df.empty:
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            col = st.selectbox("Numeric column", num_cols)
            st.write("Mean:", np.mean(df[col]))
            st.write("Median:", np.median(df[col]))
            st.write("Std Dev:", np.std(df[col]))
            st.write("Unique:", df[col].nunique())

        # Search & Delete
        scol = st.selectbox("Search Column", df.columns)
        sval = st.text_input("Search Value")
        if sval:
            st.dataframe(df[df[scol].astype(str).str.contains(sval, case=False)])
            if st.button("Delete Matches"):
                df = df[~df[scol].astype(str).str.contains(sval, case=False)]
                st.success("Deleted matching rows!")

        # Delete by Index
        if len(df) > 0:
            idx = st.number_input("Delete Row by Index", 0, len(df)-1, 0)
            if st.button("Delete Row"):
                df = df.drop(index=idx)
                st.success(f"Row {idx} deleted!")


with st.expander("⬇️ Export Data"):
    if not df.empty:
        b = io.BytesIO(); pickle.dump(df, b); b.seek(0)
        st.download_button("⬇ Pickle", b, "data.pkl")
        st.download_button("⬇ CSV", df.to_csv(index=False).encode(), "data.csv")
        st.download_button("⬇ XML", df_to_xml(df), "data.xml")
        st.download_button("⬇ TXT", df.to_string(index=False), "data.txt")
