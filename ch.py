import streamlit as st
import pandas as pd
import numpy as np
import pickle, io
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 Easy Data Dashboard", layout="wide")
st.title("📊 Easy Data Dashboard")

# ------------------------
# Helper Functions
# ------------------------
def df_to_xml(df):
    root = ET.Element('rows')
    for _, row in df.iterrows():
        r = ET.SubElement(root, 'row')
        for c in df.columns:
            ET.SubElement(r, str(c)).text = str(row[c])
    return ET.tostring(root, encoding='utf-8')

def xml_to_df(b):
    root = ET.fromstring(b)
    return pd.DataFrame([{el.tag: el.text for el in row} for row in root])

# ------------------------
# 1. Upload Data
# ------------------------
st.sidebar.header("Upload Data")
file = st.sidebar.file_uploader("Upload CSV/TXT/PKL/XML", type=["csv","txt","tsv","pkl","xml"])

df = pd.DataFrame()


if file:
    if file.name.endswith(("csv", "txt", "tsv")):
        sep = "\t" if file.name.endswith(("tsv", "txt")) else ","
        df = pd.read_csv(file, sep=sep)
    elif file.name.endswith(".pkl"):
        df = pickle.load(file)
    elif file.name.endswith(".xml"):
        df = xml_to_df(file.read())
    else:
        df = pd.DataFrame()
else:
    df = pd.DataFrame()

# Reset
if st.sidebar.button("🔄 Reset Data"):
    st.experimental_rerun()

# ------------------------
# 2. Dataset & Summary
# ------------------------
st.header("1️⃣ Dataset Preview")
st.dataframe(df)
st.write("*Shape:*", df.shape)
st.write("*Summary:*")
st.write(df.describe(include="all"))

# ------------------------
# 3. Append New Row
# ------------------------
st.header("2️⃣ Append New Row")
with st.form("append_row", clear_on_submit=True):
    new_row = {}
    for col in df.columns:
        new_row[col] = st.text_input(f"Enter {col}")
    add_btn = st.form_submit_button("Add Row")
if add_btn:
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    st.success("Row added successfully!")
    st.dataframe(df.tail())

# ------------------------
# 4. Handle Missing Values
# ------------------------
st.header("3️⃣ Handle Missing Values")
missing = df.columns[df.isnull().any()].tolist()
if missing:
    col = st.selectbox("Column with NaN", missing)
    action = st.radio("Action", ["Fill Mean","Fill Median","Fill Mode","Drop Rows","Drop Column"])
    if st.button("Apply"):
        if action=="Fill Mean": df[col].fillna(df[col].mean(), inplace=True)
        elif action=="Fill Median": df[col].fillna(df[col].median(), inplace=True)
        elif action=="Fill Mode": df[col].fillna(df[col].mode()[0], inplace=True)
        elif action=="Drop Rows": df.dropna(subset=[col], inplace=True)
        else: df.drop(columns=[col], inplace=True)
        st.success(f"{action} applied!")
else:
    st.info("No missing values found")

# ------------------------
# 5. Transformations
# ------------------------
st.header("4️⃣ Transformations & Filters")
# Convert
if not df.empty:
    c = st.selectbox("Convert column type", df.columns)
    t = st.selectbox("Convert to", ["int","float","str","datetime"])
    if st.button("Convert Type"):
        df[c] = pd.to_datetime(df[c], errors="coerce") if t=="datetime" else df[c].astype(t)
        st.success("Converted!")

# Sort
sort_cols = st.multiselect("Sort by", df.columns)
if st.button("Sort Data") and sort_cols:
    df = df.sort_values(by=sort_cols)

# Group
if st.checkbox("Group By"):
    g = st.selectbox("Group column", df.columns)
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols)>0:
        a = st.selectbox("Aggregate column", num_cols)
        f = st.selectbox("Function", ["mean","sum","count","min","max"])
        st.write(getattr(df.groupby(g)[a], f)())

# Slice
start, end = st.slider("Row Slice", 0, len(df), (0, min(len(df),10)))
st.write(df.iloc[start:end])

# Filter
expr = st.text_input("Filter expression (Ex: age > 22)")
if expr:
    try: st.dataframe(df.query(expr))
    except: st.error("Invalid expression")

# ------------------------
# 6. Visualization
# ------------------------
st.header("5️⃣ Visualization")
num_cols = df.select_dtypes(include=np.number).columns
if len(num_cols)>0:
    x = st.selectbox("X-axis", df.columns)
    y = st.selectbox("Y-axis", [c for c in num_cols if c!=x], index=0)
    chart = st.radio("Chart Type", ["Histogram","Bar","Line","Scatter"])
    if st.button("Plot Chart"):
        fig, ax = plt.subplots()
        if chart=="Histogram": ax.hist(df[x].dropna(), bins=20)
        elif chart=="Bar": ax.bar(df[x].astype(str), df[y])
        elif chart=="Line": ax.plot(df[x], df[y])
        else: ax.scatter(df[x], df[y])
        st.pyplot(fig)

# ------------------------
# 7. NumPy & Pandas Ops
# ------------------------
st.header("6️⃣ NumPy / Pandas Ops")
if len(num_cols)>0:
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

idx = st.number_input("Delete Row by Index", 0, len(df)-1, 0)
if st.button("Delete Row"):
    df = df.drop(index=idx)
    st.success(f"Row {idx} deleted!")

# ------------------------
# 8. Export
# ------------------------
st.header("7️⃣ Export Data")
b = io.BytesIO(); pickle.dump(df, b); b.seek(0)
st.download_button("⬇ Pickle", b, "data.pkl")
st.download_button("⬇ CSV", df.to_csv(index=False).encode(), "data.csv")
st.download_button("⬇ XML", df_to_xml(df), "data.xml")
st.download_button("⬇ TXT", df.to_string(index=False), "data.txt")