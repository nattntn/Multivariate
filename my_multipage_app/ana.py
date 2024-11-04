import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import chi2, f
from statsmodels.multivariate.manova import MANOVA

# ฟังก์ชันทดสอบ Multivariate Normality กรณีทราบความแปรปรวนประชากร
def multivariate_test_known_variance(X, mu, cov, alpha):
    n, p = X.shape
    X_mean = np.mean(X, axis=0)
    cov_inv = np.linalg.inv(cov)
    mahalanobis_dist = n * (X_mean - mu).T @ cov_inv @ (X_mean - mu)
    p_value = 1 - chi2.cdf(mahalanobis_dist, df=p)
    return mahalanobis_dist, p_value

# Hotelling's T² Test
def hotellings_t2(X, mu, alpha):
    n, p = X.shape
    X_mean = np.mean(X, axis=0)
    S_inv = np.linalg.inv(np.cov(X, rowvar=False))
    T2 = n * (X_mean - mu).T @ S_inv @ (X_mean - mu)
    F_value = (T2 * (n - p)) / (p * (n - 1))
    p_value = 1 - f.cdf(F_value, p, n - p)
    return T2, F_value, p_value

# Paired Sample Test สำหรับประชากรที่ไม่อิสระกัน
def paired_sample_test(X, Y, alpha):
    diff = X - Y
    n, p = diff.shape
    mean_diff = np.mean(diff, axis=0)
    S_inv = np.linalg.inv(np.cov(diff, rowvar=False))
    T2 = n * (mean_diff.T @ S_inv @ mean_diff)
    F_value = (T2 * (n - p)) / (p * (n - 1))
    p_value = 1 - f.cdf(F_value, p, n - p)
    return T2, F_value, p_value

# MANOVA Analysis
def perform_manova(df, dependent_vars, independent_var):
    df = df.rename(columns=lambda x: x.replace(' ', '_').replace('(', '').replace(')', ''))
    formula = f'{",".join(dependent_vars)} ~ {independent_var}'
    maov = MANOVA.from_formula(formula, data=df)
    return maov.mv_test()

# Main function สำหรับหน้า Analysis Page
def app():
    st.title("การทดสอบค่าเฉลี่ยกรณี 2 ประชากรขึ้นไป")

    st.write("### เกณฑ์ในการเลือกประเภทการทดสอบ:")
    st.markdown("""
    - หากคุณกำลังทดสอบ **ค่าเฉลี่ยระหว่าง 2 ประชากร**:
        - ใช้ **Multivariate Test (Known population Variance)** หรือ **Hotelling's T² Test (Unknown population Variance)** ถ้าประชากร **อิสระกัน**.
        - ใช้ **Paired Sample Test** ถ้าประชากร **ไม่อิสระกัน**.
    - หากคุณกำลังทดสอบ **ค่าเฉลี่ยในหลายประชากร**:
        - ใช้ **MANOVA** เพื่อทดสอบความแตกต่างของค่าเฉลี่ยหลายตัวแปรในหลายกลุ่ม.
    """)

    if 'df2' in st.session_state:
        df = st.session_state.df2  # ข้อมูลที่มาจากหน้า Upload หรือ Simulate
        st.write("### ข้อมูลที่ใช้ในการวิเคราะห์:")
        st.dataframe(df)

        alpha = st.number_input("กำหนดค่า p-value (ระดับนัยสำคัญ)", value=0.05, min_value=0.001, max_value=0.1, step=0.001)

        analysis_type = st.selectbox("เลือกประเภทการทดสอบ:", 
                                     ["Multivariate Test (Known Variance)", 
                                      "Hotelling's T² Test (Unknown Variance)", 
                                      "Paired Sample Test (Dependent Samples)", 
                                      "MANOVA"])

        st.write("### สมมติฐานทางสถิติ:")
        if analysis_type == "Multivariate Test (Known Variance)":
            st.write("- **$H_0$**: $\\mu_1 = \\mu_2$ (เวกเตอร์ค่าเฉลี่ยของประชากรกลุ่มที่ 1 เท่ากับ เวกเตอร์ค่าเฉลี่ยของประชากรกลุ่มที่ 2)")
            st.write("- **$H_1$**: $\\mu_1 \\neq \\mu_2$ (เวกเตอร์ค่าเฉลี่ยของประชากรกลุ่มที่ 1 ไม่เท่ากับ เวกเตอร์ค่าเฉลี่ยของประชากรกลุ่มที่ 2)")
        elif analysis_type == "Hotelling's T² Test (Unknown Variance)":
            st.write("- **$H_0$**: $\\mu_1 = \\mu_2$ (เวกเตอร์ค่าเฉลี่ยของประชากรกลุ่มที่ 1 เท่ากับ เวกเตอร์ค่าเฉลี่ยของประชากรกลุ่มที่ 2)")
            st.write("- **$H_1$**: $\\mu_1 \\neq \\mu_2$ (เวกเตอร์ค่าเฉลี่ยของประชากรกลุ่มที่ 1 ไม่เท่ากับ เวกเตอร์ค่าเฉลี่ยของประชากรกลุ่มที่ 2)")
        elif analysis_type == "Paired Sample Test (Dependent Samples)":
            st.write("- **$H_0$**: $\\mu_D = 0")
            st.write("- **$H_1$**: $\\mu_1 \\neq  0")
        elif analysis_type == "MANOVA":
            st.write("- **$H_0$**: $\\mu_1 = \\mu_2 = ... = \\mu_a )")
            st.write("- **$H_1$**: มี $\\mu_i \\neq \\mu_j$ อย่างน้อย 1 คู่ เมื่อ i \\neq j ; i=j = 1,2,...a")

        # การเลือกคอลัมน์สำหรับการทดสอบ
        if analysis_type == "Multivariate Test (Known Variance)":
            mean_input = st.text_input("ใส่ค่าเฉลี่ยของประชากร (comma-separated)", "0,0")
            mean = np.array([float(x) for x in mean_input.split(',')])

            cov_input = st.text_area("ใส่เมทริกซ์ความแปรปรวนร่วมของประชากร (comma-separated rows)", "1,0\n0,1")
            cov = np.array([[float(x) for x in row.split(',')] for row in cov_input.split('\n')])

            cols = st.multiselect("เลือกคอลัมน์ที่ต้องการศึกษา (เลือก 2 คอลัมน์)", df.select_dtypes(include=[np.number]).columns.tolist())
            if len(cols) == 2:
                if st.button("ทดสอบ Multivariate Test (Known Variance)"):
                    data_for_test = df[cols].to_numpy()
                    mahalanobis_dist, p_value = multivariate_test_known_variance(data_for_test, mean, cov, alpha)
                    st.write(f"Mahalanobis Distance: {mahalanobis_dist}")
                    st.write(f"P-value: {p_value}")
                    if p_value >= alpha:
                        st.success(f"Do not reject H0, p-value = {p_value:.4f} more than alpha {alpha}."
                                   f" Mean of {cols[0]} and {cols[1]} are similar.")
                    else:
                        st.error(f"Reject H0, p-value = {p_value:.4f} less than alpha {alpha}."
                                 f"Mean of {cols[0]} and {cols[1]} are different")
            else:
                st.warning("กรุณาเลือกคอลัมน์ 2 คอลัมน์")

        elif analysis_type == "Hotelling's T² Test (Unknown Variance)":
            mean_input = st.text_input("ใส่ค่าเฉลี่ยของประชากร (comma-separated)", "0,0")
            mean = np.array([float(x) for x in mean_input.split(',')])

            cols = st.multiselect("เลือกคอลัมน์ที่ต้องการศึกษา (เลือก 2 คอลัมน์)", df.select_dtypes(include=[np.number]).columns.tolist())
            if len(cols) == 2:
                if st.button("ทดสอบ Hotelling's T² Test"):
                    data_for_test = df[cols].to_numpy()
                    T2, F_value, p_value = hotellings_t2(data_for_test, mean, alpha)
                    st.write(f"Hotelling's T² Statistic: {T2}")
                    st.write(f"F-value: {F_value}")
                    st.write(f"P-value: {p_value}")
                    if p_value >= alpha:
                        st.success(f"Do not reject H0, p-value = {p_value:.4f} more than alpha {alpha}."
                                   f" Mean of {cols[0]} and {cols[1]} are similar.")
                    else:
                        st.error(f"Reject H0, p-value = {p_value:.4f} less than alpha {alpha}."
                                 f"Mean of {cols[0]} and {cols[1]} are different")
            else:
                st.warning("กรุณาเลือกคอลัมน์ 2 คอลัมน์")

        elif analysis_type == "Paired Sample Test (Dependent Samples)":
            cols = st.multiselect("เลือกคอลัมน์ที่ต้องการศึกษา (เลือก 2 คอลัมน์)", df.select_dtypes(include=[np.number]).columns.tolist())
            if len(cols) == 2:
                X = df[cols[0]].to_numpy()
                Y = df[cols[1]].to_numpy()

                if st.button("ทดสอบ Paired Sample Test"):
                    T2, F_value, p_value = paired_sample_test(X.reshape(-1, 1), Y.reshape(-1, 1), alpha)
                    st.write(f"T² Statistic: {T2}")
                    st.write(f"F-value: {F_value}")
                    st.write(f"P-value: {p_value}")
                    if p_value >= alpha:
                        st.success(f"Do not reject H0, p-value = {p_value:.4f} more than alpha {alpha}."
                                   f" Mean of {cols[0]} and {cols[1]} are similar.")
                    else:
                        st.error(f"Reject H0, p-value = {p_value:.4f} less than alpha {alpha}."
                                 f"Mean of {cols[0]} and {cols[1]} are different")
            else:
                st.warning("กรุณาเลือกคอลัมน์ 2 คอลัมน์")

        elif analysis_type == "MANOVA":
            dependent_vars = st.multiselect("เลือกตัวแปรตาม (numeric only)", df.select_dtypes(include=[np.number]).columns.tolist())
            independent_var = st.selectbox("เลือกตัวแปรอิสระ", df.columns.tolist())

            if dependent_vars and independent_var:
                if st.button("ทดสอบ MANOVA"):
                    result = perform_manova(df, dependent_vars, independent_var)
                    st.write(result)

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
