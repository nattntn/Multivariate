import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import chi2, f
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multivariate import test_cov_oneway
#import pingouin as pg  # เพิ่ม pingouin

def input_muVector(dim):
    st.write(f"Insert mean vector of dimension {dim}")
    muvector = np.zeros(dim)
    for i in range(dim):
        muvector[i] = st.number_input(f"Insert μ_{i+1}", value=0.0, step=0.1)
    return muvector

def input_Cov_Matrix(dim):
    st.write(f"Insert Covariance Matrix (size {dim}x{dim})")
    cov_mat = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i, dim):
            if i == j:
                cov_mat[i, j] = st.number_input(f"Variance of dimension {i+1}", value=1.0, step=0.1)
            else:
                cov_mat[i, j] = st.number_input(f"Covariance of dimensions {i+1} and {j+1}", value=0.0, step=0.1)
                cov_mat[j, i] = cov_mat[i, j]
    return cov_mat

# Check if matrix is positive semi-definite
def is_positive_semi_definite(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues >= 0)

# ฟังก์ชันทดสอบเวกเตอร์ค่าเฉลี่ยของ 1 ประชากร (กรณีทราบความแปรปรวน)
def test_one_group_known_variance(y_bar, vec_mu, cov_matrix, alpha, n):
    diff = y_bar - vec_mu
    Sigma_inv = np.linalg.inv(cov_matrix)
    Z_squared = n * np.matmul(np.matmul(diff, Sigma_inv), diff.T)
    chi2_critical = chi2.ppf(1 - alpha, len(diff))

    st.write("### สถิติที่ใช้ทดสอบ:")
    st.write(f"Statistical Test: Chi-Squared Test (χ²)")
    st.write(f"Degrees of Freedom (df): {len(diff)}")
    st.write(f"Significance Level (alpha): {alpha}")
    return Z_squared, chi2_critical

# ฟังก์ชันทดสอบเวกเตอร์ค่าเฉลี่ยของ 1 ประชากร (ไม่ทราบความแปรปรวน)
def test_one_group_unknown_variance(y_bar, vec_mu, cov_matrix, alpha, n):
    v = n - 1
    diff = y_bar - vec_mu
    p = len(diff)
    Sigma_inv = np.linalg.inv(cov_matrix)
    T_squared = n * np.matmul(np.matmul(diff, Sigma_inv), diff.T)
    F_critical = f.ppf(1 - alpha, p, v - p + 1) * ((v * p) / (v - p + 1))

    st.write("### สถิติที่ใช้ทดสอบ:")
    st.write(f"Statistical Test: F Test")
    st.write(f"Degrees of Freedom (df1, df2): ({p}, {v - p + 1})")
    st.write(f"Significance Level (alpha): {alpha}")
    return T_squared, F_critical

# ฟังก์ชันทดสอบเวกเตอร์ค่าเฉลี่ยของ 2 กลุ่มอิสระกัน (ความแปรปรวนเท่ากัน)
def test_two_groups_equal_variance(y_bar1, y_bar2, n1, n2, s1, s2, alpha):
    diff = y_bar1 - y_bar2
    p = len(diff)
    S_pool = (1 / (n1 + n2 - 2)) * ((n1 - 1) * s1 + (n2 - 1) * s2)
    S_pool_inv = np.linalg.inv(S_pool)
    T_squared = ((n1 * n2) / (n1 + n2)) * np.matmul(np.matmul(diff, S_pool_inv), diff.T)
    F_critical = f.ppf(1 - alpha, p, n1 + n2 - p - 1)

    st.write("### สถิติที่ใช้ทดสอบ:")
    st.write(f"Statistical Test: F Test")
    st.write(f"Degrees of Freedom (df1, df2): ({p}, {n1 + n2 - p - 1})")
    st.write(f"Significance Level (alpha): {alpha}")
    return T_squared, F_critical

# ฟังก์ชันทดสอบเวกเตอร์ค่าเฉลี่ยของ 2 กลุ่มอิสระกัน (ความแปรปรวนไม่เท่ากัน)
def test_two_groups_unequal_variance(y_bar1, y_bar2, s1, s2, n1, n2, alpha):
    diff = y_bar1 - y_bar2
    S1_inv = np.linalg.inv(s1)
    S2_inv = np.linalg.inv(s2)
    pooled_inv = (S1_inv / n1) + (S2_inv / n2)
    T_squared = np.matmul(np.matmul(diff.T, pooled_inv), diff)
    F_critical = f.ppf(1 - alpha, len(diff), min(n1, n2) - len(diff))

    st.write("### สถิติที่ใช้ทดสอบ:")
    st.write(f"Statistical Test: F Test")
    st.write(f"Degrees of Freedom (df1, df2): ({len(diff)}, {min(n1, n2) - len(diff)})")
    st.write(f"Significance Level (alpha): {alpha}")
    return T_squared, F_critical

# ฟังก์ชันทดสอบค่าเฉลี่ย 2 กลุ่ม ที่ไม่อิสระกัน
def test_paired_samples(X, Y, alpha):
    diff = X - Y
    n, p = diff.shape
    mean_diff = np.mean(diff, axis=0)
    cov_inv = np.linalg.inv(np.cov(diff, rowvar=False))
    T2 = n * (mean_diff.T @ cov_inv @ mean_diff)
    F_value = (T2 * (n - p)) / (p * (n - 1))
    F_critical = f.ppf(1 - alpha, p, n - p)

    st.write("### สถิติที่ใช้ทดสอบ:")
    st.write(f"Statistical Test: F Test")
    st.write(f"Degrees of Freedom (df1, df2): ({p}, {n - p})")
    st.write(f"Significance Level (alpha): {alpha}")
    return T2, F_critical

# MANOVA Analysis
def perform_manova(df, dependent_vars, independent_var):
    # แก้ไขชื่อคอลัมน์โดยการลบสัญลักษณ์พิเศษและช่องว่างออก
    df = df.rename(columns=lambda x: x.replace(' ', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', ''))
    dependent_vars = [var.replace(' ', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '') for var in dependent_vars]
    independent_var = independent_var.replace(' ', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '')

    # แก้ไขสูตรการวิเคราะห์ MANOVA
    formula = f"{'+'.join(dependent_vars)} ~ {independent_var}"
    maov = MANOVA.from_formula(formula, data=df)
    return maov.mv_test()

# ฟังก์ชันทดสอบ Box's M Test โดยใช้ test_cov_oneway
def perform_box_m_test(cov_list, nobs_list, alpha):
    test_result = test_cov_oneway(cov_list, nobs_list)
    
    # Access the correct attributes
    m_stat = test_result.statistic
    p_value = test_result.pvalue
    return m_stat, p_value


# ฟังก์ชันทดสอบ Box's M Test โดยใช้ test_cov_oneway
#def perform_box_m_test(cov_list, nobs_list, alpha):
#    test_result = test_cov_oneway(cov_list, nobs_list)
#    m_stat = test_result.statistic
#    p_value = test_result.pvalue
#    F_value = test_result.f_value
#    F_critical = test_result.f_critical
#    chi2_critical = test_result.chi2_critical
#    df1 = test_result.df_denom
#    df2 = test_result.df_num
#    return m_stat, p_value, F_value, F_critical, chi2_critical, df1, df2

# Main function สำหรับหน้า Analysis Page
def app():
    st.title("การทดสอบสมมติฐานในหลายประชากร")

    st.write("### เกณฑ์ในการเลือกประเภทการทดสอบ:")
    st.markdown("""
    - หากคุณกำลังทดสอบ **ค่าเฉลี่ยของ 1 กลุ่ม**:
        - ใช้ **One Group Test (Known Variance)** หากทราบเมทริกซ์ความแปรปรวน.
        - ใช้ **One Group Test (Unknown Variance)** หากไม่ทราบเมทริกซ์ความแปรปรวน.
    - หากคุณกำลังทดสอบ **ค่าเฉลี่ยระหว่าง 2 กลุ่มอิสระกัน**:
        - ใช้ **Two Group Test (Equal Variance)** หากความแปรปรวนเท่ากัน.
        - ใช้ **Two Group Test (Unequal Variance)** หากความแปรปรวนไม่เท่ากัน.
    - หากคุณกำลังทดสอบ **ค่าเฉลี่ยในสองกลุ่มไม่อิสระกัน**:
        - ใช้ **Paired Samples Test** สำหรับกลุ่มที่ไม่อิสระกัน.
    - หากคุณกำลังทดสอบ **ค่าเฉลี่ยในหลายกลุ่ม**:
        - ใช้ **MANOVA** เพื่อทดสอบความแตกต่างของค่าเฉลี่ยหลายตัวแปรในหลายกลุ่ม.
    """)

    if 'df2' in st.session_state:
        df = st.session_state.df2  # ข้อมูลที่มาจากหน้า Upload หรือ Simulate
        st.write("### ข้อมูลที่ใช้ในการวิเคราะห์:")
        st.dataframe(df)

        alpha = st.number_input("กำหนดค่า p-value (ระดับนัยสำคัญ)", value=0.05, min_value=0.001, max_value=0.1, step=0.001)

        analysis_type = st.selectbox("เลือกประเภทการทดสอบ:", 
                                     ["One Group Test (Known Variance)", 
                                      "One Group Test (Unknown Variance)", 
                                      "Two Group Test (Equal/Unequal Variance)", 
                                      "Paired Samples Test", 
                                      "MANOVA"])
        
        if analysis_type == "One Group Test (Known Variance)":
            st.write("### สมมติฐานทางสถิติ:")
            st.latex(r"H_0: \mu = \mu_0 \quad \text{vs.} \quad H_1: \mu \neq \mu_0")
            cols = st.multiselect("เลือกคอลัมน์ที่ต้องการศึกษา", df.select_dtypes(include=[np.number]).columns.tolist())
            dim = len(cols)
            
            if dim >= 2:
                st.write("### Insert Mean Vector")
                vec_mu = input_muVector(dim)
                st.write("Mean vector:")
                st.table(pd.DataFrame(vec_mu.reshape(1, -1), columns=[f"μ_{i+1}" for i in range(dim)]))
                
                st.write("### Insert Covariance Matrix")
                cov_matrix = input_Cov_Matrix(dim)
                st.write("Covariance Matrix:")
                st.table(pd.DataFrame(cov_matrix, columns=[f"Dim_{i+1}" for i in range(dim)], index=[f"Dim_{i+1}" for i in range(dim)]))

                if not is_positive_semi_definite(cov_matrix):
                    st.error("The covariance matrix is not positive semi-definite. Please provide a valid matrix.")
                else:
                    st.success("The covariance matrix is positive semi-definite.")
                    
                    vec_y_bar = df[cols].mean().to_numpy()
                    n = df.shape[0]
                    if st.button("ทดสอบ One Group Test (Known Variance)"):
                        Z_squared, chi2_critical = test_one_group_known_variance(vec_y_bar, vec_mu, cov_matrix, alpha, n)
                        st.write(f"Z² = {Z_squared:.4f}")
                        st.write(f"Chi² critical value = {chi2_critical:.4f}")
                        if Z_squared > chi2_critical:
                            st.error(f"Reject H0, Z² = {Z_squared:.4f} > Chi² critical = {chi2_critical:.4f}")
                        else:
                            st.success(f"Do not reject H0, Z² = {Z_squared:.4f} <= Chi² critical = {chi2_critical:.4f}")

        elif analysis_type == "One Group Test (Unknown Variance)":
            st.write("### สมมติฐานทางสถิติ:")
            st.latex(r"H_0: \mu = \mu_0 \quad \text{vs.} \quad H_1: \mu \neq \mu_0")
            cols = st.multiselect("เลือกคอลัมน์ที่ต้องการศึกษา", df.select_dtypes(include=[np.number]).columns.tolist())
            dim = len(cols)
            
            if dim >= 2:
                st.write("### Insert Mean Vector")
                vec_mu = input_muVector(dim)
                st.write("Mean vector:")
                st.table(pd.DataFrame(vec_mu.reshape(1, -1), columns=[f"μ_{i+1}" for i in range(dim)]))

                vec_y_bar = df[cols].mean().to_numpy()
                cov_matrix = df[cols].cov().to_numpy()
                st.write(f"เวกเตอร์ค่าเฉลี่ยของตัวอย่าง (y_bar): {vec_y_bar}")
                st.write(f"เมทริกซ์ความแปรปรวนร่วมของตัวอย่าง (S):")
                st.table(pd.DataFrame(cov_matrix, columns=[f"Dim_{i+1}" for i in range(dim)], index=[f"Dim_{i+1}" for i in range(dim)]))
                n = df.shape[0]
                
                if st.button("ทดสอบ One Group Test (Unknown Variance)"):
                    T_squared, F_critical = test_one_group_unknown_variance(vec_y_bar, vec_mu, cov_matrix, alpha, n)
                    st.write(f"T² = {T_squared:.4f}")
                    st.write(f"F critical value = {F_critical:.4f}")
                    if T_squared > F_critical:
                        st.error(f"Reject H0, T² = {T_squared:.4f} > F critical = {F_critical:.4f}")
                    else:
                        st.success(f"Do not reject H0, T² = {T_squared:.4f} <= F critical = {F_critical:.4f}")

        elif analysis_type == "Two Group Test (Equal/Unequal Variance)":
            st.write("### สมมติฐานทางสถิติ:")
            st.latex(r"H_0: \mu_1 = \mu_2 \quad \text{vs.} \quad H_1: \mu_1 \neq \mu_2")

            category_col = st.selectbox("เลือกคอลัมน์ที่เป็นประเภทสำหรับแยกกลุ่ม", df.columns.tolist())
            unique_values = df[category_col].unique()

            if len(unique_values) < 2:
                st.error("คอลัมน์ที่เลือกต้องมีค่าอย่างน้อย 2 ค่าในการแยกกลุ่ม")
            else:
                selected_values = st.multiselect(f"เลือก 2 ค่าในคอลัมน์ '{category_col}' ที่ต้องการศึกษา", unique_values, default=unique_values[:2])
                
                if len(selected_values) != 2:
                    st.error("กรุณาเลือกค่าที่ต้องการศึกษาให้ครบ 2 ค่า")
                else:
                    filtered_df = df[df[category_col].isin(selected_values)]
                    st.write("ข้อมูลที่กรองแล้ว:")
                    st.dataframe(filtered_df)

                    # ให้ผู้ใช้เลือกคอลัมน์ที่ต้องการศึกษาเวกเตอร์ค่าเฉลี่ย
                    cols = st.multiselect("เลือกคอลัมน์ที่ต้องการศึกษาเวกเตอร์ค่าเฉลี่ย", filtered_df.select_dtypes(include=[np.number]).columns.tolist())
                    st.write("#### Test of homogeneous covariance matrices")
                    #st.latex(r"H_0: \mu_1 = \mu_2 \quad \text{vs.} \quad H_1: \mu_1 \neq \mu_2")
                    st.latex(r"H_0: \Sigma_{1} = \Sigma_{2} \quad \text{vs.} \quad H_1: \Sigma_{1} \neq \Sigma_{2}")
                    if len(cols) >= 2:
                        # แบ่งข้อมูลเป็นสองกลุ่ม
                        group_1 = filtered_df[filtered_df[category_col] == selected_values[0]]
                        group_2 = filtered_df[filtered_df[category_col] == selected_values[1]]

                        vec_y_bar1 = group_1[cols].mean().to_numpy()
                        vec_y_bar2 = group_2[cols].mean().to_numpy()

                        n1 = group_1.shape[0]
                        n2 = group_2.shape[0]

                        s1 = group_1[cols].cov().to_numpy()
                        s2 = group_2[cols].cov().to_numpy()

                        cov_list = [s1, s2]
                        nobs_list = [n1, n2]

                        # ทดสอบ Box's M Test พร้อมการแสดงค่า F-test และ Chi-square
                        m_stat, p_value = perform_box_m_test(cov_list, nobs_list, alpha)# แบ่งข้อมูลเป็นสองกลุ่ม
                        st.write(f"### ผลการทดสอบ Box's M Test:")
                        st.write(f"- Box's M Statistic: {m_stat:.4f}")
                        st.write(f"- P-value: {p_value:.4f}")

                        
                        if p_value < alpha:
                            st.error(f"Reject H0, p-value = {p_value:.4f} < alpha = {alpha}. Thus,  𝞢_{1} != 𝞢_{2}")
                            # ทดสอบแบบความแปรปรวนไม่เท่ากัน
                            st.write("#### two groups unequal variance")
                            T_squared, F_critical = test_two_groups_unequal_variance(vec_y_bar1, vec_y_bar2, s1, s2, n1, n2, alpha)
                            st.write(f"T² = {T_squared:.4f}")
                            st.write(f"F critical value = {F_critical:.4f}")
                            if T_squared > F_critical:
                                st.error(f"Reject H0, T² = {T_squared:.4f} > F critical = {F_critical:.4f}. Thus, mu_1 != mu_2")
                            else:
                                st.success(f"Do not reject H0, T² = {T_squared:.4f} <= F critical = {F_critical:.4f}. Thus, mu_1 = mu_2")
                        else:
                            st.success(f"Do not reject H0, p-value = {p_value:.4f} >= alpha = {alpha}. Thus, 𝞢_{1} = 𝞢_{2}")
                            # ทดสอบแบบความแปรปรวนเท่ากัน
                            st.write("#### two groups equal variance")
                            T_squared, F_critical = test_two_groups_equal_variance(vec_y_bar1, vec_y_bar2, n1, n2, s1, s2, alpha)
                            st.write(f"T² = {T_squared:.4f}")
                            st.write(f"F critical value = {F_critical:.4f}")
                            if T_squared > F_critical:
                                st.error(f"Reject H0, T² = {T_squared:.4f} > F critical = {F_critical:.4f}. Thus, mu_1 != mu_2")
                            else:
                                st.success(f"Do not reject H0, T² = {T_squared:.4f} <= F critical = {F_critical:.4f}. Thus, mu_1 = mu_2")

        elif analysis_type == "Paired Samples Test":
            st.write("### สมมติฐานทางสถิติ:")
            st.latex(r"H_0: \mu_D = 0 \quad \text{vs.} \quad H_1: \mu_D \neq 0")
            cols = st.multiselect("เลือกคอลัมน์ที่ต้องการศึกษา", df.select_dtypes(include=[np.number]).columns.tolist())
            if len(cols) == 2:
                X = df[cols[0]].to_numpy()
                Y = df[cols[1]].to_numpy()
                if st.button("ทดสอบ Paired Samples Test"):
                    T2, F_critical = test_paired_samples(X.reshape(-1, 1), Y.reshape(-1, 1), alpha)
                    st.write(f"T² = {T2:.4f}")
                    st.write(f"F critical value = {F_critical:.4f}")
                    if T2 > F_critical:
                        st.error(f"Reject H0, T² = {T2:.4f} > F critical = {F_critical:.4f}")
                    else:
                        st.success(f"Do not reject H0, T² = {T2:.4f} <= F critical = {F_critical:.4f}")

        elif analysis_type == "MANOVA":
            st.write("### สมมติฐานทางสถิติ:")
            st.latex(r"H_0: \mu_1 = \mu_2 = ... = \mu_a \quad \text{vs.} \quad H_1: \mu_i \neq \mu_j")
            
            # ให้ผู้ใช้เลือกคอลัมน์ที่ใช้สำหรับแยกกลุ่ม (สามารถเลือกได้ทั้งตัวเลขหรือประเภท class)
            category_col = st.selectbox("เลือกคอลัมน์สำหรับแบ่ง class", df.columns.tolist())
            unique_values = df[category_col].unique()

            if len(unique_values) < 3:
                st.error("คอลัมน์ที่เลือกต้องมีค่าอย่างน้อย 3 ค่าในการแบ่งกลุ่ม")
            else:
                # ให้ผู้ใช้เลือกมากกว่า 2 ค่าขึ้นไปที่ต้องการศึกษา
                selected_values = st.multiselect(f"เลือกค่าต่างๆ ในคอลัมน์ '{category_col}' ที่ต้องการศึกษา (เลือกมากกว่า 2 ค่า)", unique_values, default=unique_values[:3])
                
                if len(selected_values) < 2:
                    st.error("กรุณาเลือกค่าที่ต้องการศึกษาให้มากกว่า 2 ค่า")
                else:
                    # กรองข้อมูลให้เหลือเฉพาะค่าที่เลือก
                    filtered_df = df[df[category_col].isin(selected_values)]
                    st.write("ข้อมูลที่กรองแล้ว:")
                    st.dataframe(filtered_df)

                    # ให้ผู้ใช้เลือกตัวแปรตาม (dependent variables)
                    dependent_vars = st.multiselect("เลือกตัวแปรที่ใช้ในการศึกษา (numeric only)", filtered_df.select_dtypes(include=[np.number]).columns.tolist())
                    independent_var = category_col

                    if dependent_vars:
                        # ดึงข้อมูลเพื่อใช้ทดสอบ M Test
                        group_1 = filtered_df[filtered_df[category_col] == selected_values[0]]
                        group_2 = filtered_df[filtered_df[category_col] == selected_values[1]]

                        cov_list = [group_1[dependent_vars].cov().to_numpy(), group_2[dependent_vars].cov().to_numpy()]
                        nobs_list = [len(group_1), len(group_2)]

                        # ทดสอบ M-test ก่อน
                        
                        st.write("#### Test of homogeneous covariance matrices")
                        st.latex(r"H_0: \Sigma_{1} = \Sigma_{2} = ... = \Sigma_{g} \quad \text{vs.} \quad H_1: at least two \Sigma_{i}'s are different ")
                        m_stat, p_value = perform_box_m_test(cov_list, nobs_list, alpha)

                        st.write("##### ผลการทดสอบ Box's M Test")
                        st.write(f"Box's M statistic: {m_stat:.4f}")
                        st.write(f"P-value: {p_value:.4f}")

                        if p_value < alpha:
                            st.error(f"Reject H0 (M Test failed): p-value = {p_value:.4f} < alpha = {alpha}. Thus, at least two Σ_i's are different")
                        else:
                            st.success(f"Do not reject H0 (M Test passed): p-value = {p_value:.4f} >= alpha = {alpha}. Thus, Σ_1 = Σ_2 = ... = Σ_g. Proceeding to MANOVA.")
                            # เมื่อผ่าน M-test แล้วจึงทำการทดสอบ MANOVA ต่อไป
                            if st.button("ทดสอบ MANOVA"):
                                result = perform_manova(filtered_df, dependent_vars, independent_var)
                                st.write(result)
                                st.write("### สรุปผลการทดสอบ MANOVA")
                                for key, test_result in result.results.items():
                                    p_value = test_result['stat']['Pr > F'][0]
                                    if p_value < alpha:
                                        st.error(f"Reject H0 for {key} (p-value = {p_value:.4f} < alpha = {alpha})")
                                    else:
                                        st.success(f"Do not reject H0 for {key} (p-value = {p_value:.4f} >= alpha = {alpha})")


    else:
        st.warning("ไม่พบข้อมูลสำหรับการวิเคราะห์ กรุณาอัปโหลดหรือจำลองข้อมูลใหม่.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Upload Page"):
                st.session_state.page = "Upload File"
                st.experimental_rerun()
        with col2:
            if st.button("Go to Simulate Page"):
                st.session_state.page = "Simulate Data"
                st.experimental_rerun()
