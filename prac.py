import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, os
from dicttoxml import dicttoxml

st.set_page_config(page_title="Student Management System", layout="wide")


st.sidebar.title("Load Data")
file = st.sidebar.file_uploader("Upload CSV", type="csv")
if file: 
    df = pd.read_csv(file)
    st.success("CSV Loaded Successfully")
else:
    st.warning("Please upload a CSV file to proceed.")

if 'df' in locals():
    
    st.write("### Data Cleaning")

   
    if st.checkbox("Remove Duplicate Rows"):
        before = len(df)
        df.drop_duplicates(inplace=True)
        after = len(df)
        st.success(f"Removed {before - after} duplicate rows")

   
    if st.checkbox("Fill Missing Values"):
        
        df.fillna(df.mean(numeric_only=True), inplace=True)
       
        for col in df.select_dtypes(include="object").columns:
            df[col].fillna("Unknown", inplace=True)
        st.success("Missing values filled")

    
    st.title("📚 Student Management & Performance Dashboard")
    st.write("Domain: Student data (marks in Maths, Science, English).")
    
    st.markdown("---")
    st.subheader("Dataset Overview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.json({"Columns": list(df.columns), "Missing values": df.isnull().sum().to_dict()})

    
    st.markdown("---")
    st.subheader("Save / Download Dataset")
    os.makedirs("exports", exist_ok=True)

    
    csv_file = "exports/students.csv"
    if st.button("Save as CSV"):
        df.to_csv(csv_file, index=False)
        st.success("Saved CSV")
    st.download_button("Download CSV", df.to_csv(index=False), file_name="students.csv", mime="text/csv")

    
    if st.button("Append to CSV"):
        df.to_csv(csv_file, mode="a", index=False, header=False)
        st.success("Appended CSV")

    
    pkl_file = "exports/students.pkl"
    if st.button("Save as Pickle"):
        with open(pkl_file, "wb") as f: pickle.dump(df, f)
        st.success("Saved Binary (Pickle)")
    st.download_button("Download Pickle", pickle.dumps(df), file_name="students.pkl", mime="application/octet-stream")

  
    xml_file = "exports/students.xml"
    if st.button("Save as XML"):
        xml_data = dicttoxml(df.to_dict("records"), custom_root="students", attr_type=False)
        with open(xml_file, "wb") as f: f.write(xml_data)
        st.success("Saved XML")
    st.download_button("Download XML", dicttoxml(df.to_dict("records"), custom_root="students", attr_type=False), 
                       file_name="students.xml", mime="application/xml")

   
    st.markdown("---")
    st.subheader("Analysis & Visualization")
    st.write("Summary Stats:")
    st.dataframe(df.describe())

    
    st.write("### Data Manipulations")
    if st.checkbox("Convert Maths to float type"):
        df["Maths"] = df["Maths"].astype(float)
        st.write(df.dtypes)

  
    sort_column = st.selectbox("Sort by column:", df.columns.tolist())
    if st.button("Sort Data"):
        st.dataframe(df.sort_values(by=sort_column, ascending=False))

   
    st.write("Slice Rows by index")
    start_idx = st.number_input("Start index:", min_value=0, max_value=len(df)-1, value=0)
    end_idx = st.number_input("End index:", min_value=0, max_value=len(df), value=5)
    if st.button("Show Slice"):
        st.dataframe(df.iloc[start_idx:end_idx])

   
    if "Class" in df.columns:
        selected_class = st.selectbox("Filter by Class:", df["Class"].unique())
        if st.button("Filter Class"):
            st.dataframe(df.loc[df["Class"]==selected_class])

   
    st.markdown("---")
    st.subheader("Visualization")
    plot_option = st.selectbox("Choose Plot", ["Histogram (Maths)", "Bar (Avg by Class)", "Scatter (Maths vs Science)"])
    
    if plot_option == "Histogram (Maths)":
        fig, ax = plt.subplots()
        ax.hist(df["Maths"], bins=10, color="skyblue", edgecolor="black")
        ax.set_title("Maths Marks Distribution")
        ax.set_xlabel("Marks")
        ax.set_ylabel("Number of Students")
        st.pyplot(fig)
        
    elif plot_option == "Bar (Avg by Class)":
        if "Class" in df.columns:
            avg = df.groupby("Class")[["Maths","Science","English"]].mean()
            st.bar_chart(avg)
            
    else:  
        fig, ax = plt.subplots()
        ax.scatter(df["Maths"], df["Science"], alpha=0.6, color="green")
        ax.set_xlabel("Maths")
        ax.set_ylabel("Science")
        ax.set_title("Maths vs Science")
        st.pyplot(fig)

   
    st.markdown("---")
    st.subheader("Interactive Queries")
    
    if st.button("Top 5 Students"):
        st.dataframe(df.nlargest(5, "Total")[["Name","Class","Total"]])

    if st.button("Pass/Fail Count (>=40)"):
        passed_mask = (df[["Maths","Science","English"]] >= 40).all(axis=1)
        passed_count = int(passed_mask.sum())
        failed_count = int(len(df) - passed_count)
        st.write({"Passed": passed_count, "Failed": failed_count})
        failed_students = df.loc[~passed_mask, ["Name","Class","Maths","Science","English","Total"]]
        if not failed_students.empty:
            st.subheader("Failed Students")
            st.dataframe(failed_students)
        else:
            st.success("No students failed.")

    subject = st.selectbox("Average marks in subject:", ["Maths","Science","English"])
    if st.button("Show Average"):
        st.write(f"Average {subject} marks:", df[subject].mean())

  
    st.markdown("---")
    st.subheader("Extra Queries")
    if st.button("Min/Max Marks per Subject"):
        st.write(df[["Maths","Science","English"]].agg(["min","max"]))

    if st.button("Students Failed in Maths"):
        failed_maths = df.loc[df["Maths"]<40, ["Name","Class","Maths"]]
        st.dataframe(failed_maths)


