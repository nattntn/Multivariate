import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.datasets import load_iris, load_wine, load_diabetes
from pingouin import multivariate_normality

def p_value():
    colss = st.columns(1)
    p_value = colss[0].number_input("P-values", value=None, placeholder="Type a number...")
    return p_value

def test_multivariate_normality(group,alpha):
    st.write(f"Number of samples: {group.shape[0]}")

    vec_mu = group.mean().values
    mat_cov = group.cov().values
    sample_size = group.shape[0]
    
    st.write('Mean of each attribute:', vec_mu)
    st.write('Variance and Covariance of each attribute:', mat_cov)
    st.write(f'Number of samples: {sample_size}')

    samples = np.random.multivariate_normal(vec_mu, mat_cov, int(sample_size))

    # ทดสอบการแจกแจงแบบหลายตัวแปรด้วย Mardia's test
    res = multivariate_normality(samples, alpha=alpha)
    st.write(f'P-value: {res.pval}')
    
    # แสดงผลลัพธ์แบบเด่นชัด
    if res.pval >= alpha:
        st.success(f'Do not reject H0. Population distributes the multivariate normal distribution (p-value = {res.pval:.4f})', icon="✅")
        return True
    else:
        st.error(f'Reject H0. Population does not distribute the multivariate normal distribution (p-value = {res.pval:.4f})', icon="❌")
        return False


def app():
    st.title("Upload File or Use Preloaded Datasets")

    # ให้ผู้ใช้เลือกว่าจะอัปโหลดไฟล์หรือใช้ dataset ที่เตรียมไว้
    data_option = st.radio(
        "Choose a data source",
        ("Upload your own file", "Use preloaded dataset (Iris, Wine, Diabetes)")
    )
    
    if data_option == "Upload your own file":
        uploaded_file = st.file_uploader("เลือกไฟล์ CSV หรือ Excel", type=["csv", "xlsx"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
    else:
        dataset_name = st.selectbox("Select a dataset", ["Iris", "Wine", "Diabetes"])
        
        if dataset_name == "Iris":
            st.write("""
            ### Iris Dataset
            The Iris dataset contains 150 samples of iris flowers, with three species: Iris setosa, Iris virginica, and Iris versicolor.
            Each sample has four features: sepal length, sepal width, petal length, and petal width.
            """)
            data = load_iris()

        elif dataset_name == "Wine":
             st.write("""
             ### Wine Dataset
             The Wine dataset contains the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars.
             There are 178 samples with 13 continuous features, which describe various chemical properties such as alcohol content, malic acid, ash, and more.
             The dataset is often used for classification tasks.
             """)
             data = load_wine()
             
        elif dataset_name == "Diabetes":
            st.write("""
            ### Diabetes Dataset
            The Diabetes dataset contains 442 diabetes patients, with 10 baseline variables used to predict disease progression after one year.
            This dataset is often used for regression tasks in the medical field.
            """)
            data = load_diabetes()

        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        st.write(f"Loaded {dataset_name} dataset")

    # ตรวจสอบว่ามีข้อมูลอยู่หรือไม่
    if 'df' in locals():
        st.write("ข้อมูลในไฟล์:")
        st.dataframe(df)
        st.write(f'The number of records: {df.shape[0]}')
        st.write(f'The number of columns: {df.shape[1]}')

        # การใช้ ydata-profiling
        st.write("## Data Profiling Report")
        profile = ProfileReport(df, explorative=True)
        st_profile_report(profile)

        if st.button('Data cleansing'):
            df2 = df.select_dtypes(include=['int', 'float'])
            st.write("**Note**: Only numeric data will be used for analysis.")
            st.write('Filter numeric table')
            st.dataframe(df2)
            st.write(f'The number of records: {df2.shape[0]}')
            st.write(f'The number of columns: {df2.shape[1]}')

            st.session_state.df2 = df2
            st.session_state.cols_with_missing = df2.columns[df2.isnull().any()].tolist()
            st.session_state.step = 1

    # การจัดการข้อมูล
    if 'step' in st.session_state and st.session_state.step == 1:
        df2 = st.session_state.df2
        cols_with_missing = st.session_state.cols_with_missing

        if cols_with_missing:
            st.write("Columns with missing values:")
            st.write(cols_with_missing)

            selected_cols = st.multiselect("Select columns to manage missing values", cols_with_missing)

            if selected_cols:
                st.session_state.selected_cols = selected_cols
                st.session_state.step = 2
        else:
            st.write("No missing values found. Proceeding to normality test.")  # ข้อความเมื่อไม่มี missing values
            st.session_state.step = 3  # ข้ามไปทดสอบการแจกแจงปกติ

    if 'step' in st.session_state and st.session_state.step == 2:
        df2 = st.session_state.df2
        selected_cols = st.session_state.selected_cols

        action = st.radio("Choose an action", ("Fill missing values", "Drop rows with missing values"))

        if action == "Fill missing values":
            fill_method = st.radio("Choose a fill method", ("Fill with specific value", "Fill with mean"))

            if fill_method == "Fill with specific value":
                fill_value = st.text_input("Enter value to fill missing values")
                if st.button("Apply Fill Missing Values"):
                    if fill_value:
                        df2[selected_cols] = df2[selected_cols].fillna(fill_value)
                        st.session_state.df2 = df2
                        st.write("Updated DataFrame after filling missing values:")
                        st.dataframe(df2)
                        st.session_state.step = 3
                    else:
                        st.write("Please enter a value to fill missing values.")

            elif fill_method == "Fill with mean":
                if st.button("Apply Fill Missing Values with Mean"):
                    df2[selected_cols] = df2[selected_cols].apply(lambda col: col.fillna(col.mean()))
                    st.session_state.df2 = df2
                    st.write("Updated DataFrame after filling missing values with mean:")
                    st.dataframe(df2)
                    st.session_state.step = 3

        elif action == "Drop rows with missing values":
            if st.button("Apply Drop Rows with Missing Values"):
                df2 = df2.dropna(subset=selected_cols)
                st.session_state.df2 = df2
                st.write("Updated DataFrame after dropping rows with missing values:")
                st.dataframe(df2)
                st.session_state.step = 3

    # ทดสอบการแจกแจงแบบปกติ
    if 'step' in st.session_state and st.session_state.step == 3:
        st.write("Final updated DataFrame:")
        st.dataframe(st.session_state.df2)

        # การทดสอบการแจกแจงแบบ Multivariate Normal Distribution
        st.write("### การทดสอบการแจกแจงแบบ Multivariate Normal Distribution")
        # ทดสอบการแจกแจงปกติหลายตัวแปรด้วย Mardia's Test
        st.write("#### สถิติทดสอบที่ใช้ใน `multivariate_normality()` คือ **Mardia's Test**")
        st.markdown("""
        - **Mardia's Skewness Test**: ทดสอบความสมมาตร (Skewness) ของข้อมูลหลายตัวแปร.
        - **Mardia's Kurtosis Test**: ทดสอบค่าความนูน (Kurtosis) ของข้อมูลหลายตัวแปร.
        """)

        numeric_columns = st.session_state.df2.select_dtypes(include=['int', 'float']).columns.tolist()
        selected_cols_for_test = st.multiselect("Select columns for normal distribution test (only include numeric types)", numeric_columns, default=numeric_columns)

        if selected_cols_for_test and len(selected_cols_for_test) >= 2:
            df_for_test = st.session_state.df2[selected_cols_for_test]
            # แบ่งกลุ่มตามคอลัมน์ที่เป็น categorical
            group_col = st.selectbox("Select a categorical column to group by (optional)", [None] + df.columns.tolist())

            # รับค่า p-value เพียงครั้งเดียวเพื่อใช้กับทุกกลุ่ม
            alpha = st.number_input("Insert your p-value (significance level)", value=0.05, min_value=0.001, max_value=0.1, step=0.001)
            st.write(f"We accept H0 if p-values >= {alpha}")
            st.write("H0: Population distributes the multivariate normal distribution")
            st.write("Ha: Population does not distribute the multivariate normal distribution")

            all_groups_normal = []  # เก็บผลการทดสอบของแต่ละกลุ่ม
            
            
            if group_col:
                groups = df_for_test.groupby(df[group_col])
                for name, group in groups:
                    st.write(f"Testing group: {name}")
                    is_normal = test_multivariate_normality(group, alpha)
                    all_groups_normal.append(is_normal)
            else:
                st.write("Testing entire dataset as a single group")
                is_normal = test_multivariate_normality(df_for_test, alpha)
                all_groups_normal.append(is_normal)

            # ถ้าผลการทดสอบของทุกกลุ่มเป็น True แสดงปุ่มไปหน้าวิเคราะห์
            if all(all_groups_normal):
                st.success("All groups distribute the multivariate normal distribution. Proceed to analysis.")
                if st.button("Go to Analysis Page"):
                    st.write("Navigating to the analysis page...")
                    st.session_state.page = "Analysis"
            else:
                st.error("Not all groups distribute the multivariate normal distribution. Please upload new data.")
