import streamlit as st
import numpy as np
from scipy.stats import multivariate_normal
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd

# Function to plot 3D surface plot for Bivariate case
def my_plot_mvn_3d_surface(X, Y, pdf):
    fig = go.Figure(data=[go.Surface(z=pdf, x=X, y=Y, showscale=True)])
    fig.update_layout(title="3D Surface Plot of Bivariate Normal Distribution", autosize=False,
                      width=700, height=700,
                      margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)

# Function to plot Contour Plot
def my_plot_contour(pdf, X, Y):
    fig_2, ax = plt.subplots()
    contour = ax.contour(X, Y, pdf)
    ax.set_title('Contour Plot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    st.pyplot(fig_2)

# Input function for mean vector
def input_muVector(dim):
    st.write(f"Insert mean vector of dimension {dim}")
    muvector = np.zeros(dim)
    for i in range(dim):
        muvector[i] = st.number_input(f"Insert μ_{i+1}", value=0.0, placeholder="Type a number...", step=0.1)
    return muvector 

# Input function for covariance matrix
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

# Input function for number of samples
def n_sample():
    return st.number_input("Number of samples", value=100, min_value=1, step=1)

def app():
    st.title("Multivariate Normal Distribution Simulator")
    st.write("""
    ## ข้อกำหนดในการใช้งาน
    1. **การสร้างข้อมูลจากการแจกแจงแบบปกติหลายมิติ (Multivariate Normal Distribution)**: คุณจะได้จำลองข้อมูลที่มีการแจกแจงแบบปกติหลายมิติ.
    2. **เวกเตอร์ค่าเฉลี่ย (Mean Vector)**: คุณต้องป้อนเวกเตอร์ค่าเฉลี่ยสำหรับแต่ละมิติ.
    3. **เมทริกซ์ความแปรปรวนร่วม (Covariance Matrix)**: เมทริกซ์ต้องเป็นเมทริกซ์สมบูรณ์ (Positive Semi-Definite).
    4. **จำนวนมิติ (Dimensions)**: เลือกจำนวนมิติที่ต้องการสร้างได้ตั้งแต่ 2 มิติขึ้นไป.
    5. **จำนวนตัวอย่าง (Sample Size)**: เลือกจำนวนตัวอย่างที่ต้องการสร้างได้.
    6. **ทดสอบการแจกแจงแบบปกติ (Normality Test)**: สามารถทำการทดสอบการแจกแจงแบบปกติสำหรับกรณี Bivariate.
    """)

    # Let the user choose the number of dimensions
    dim = st.number_input("Select the number of dimensions", value=2, min_value=2, step=1)

    # Get the mean vector and covariance matrix from user input
    vec_mu = input_muVector(dim)
    st.write("Mean vector:")
    st.table(pd.DataFrame(vec_mu.reshape(1, -1), columns=[f"μ_{i+1}" for i in range(dim)]))

    mat_cov = input_Cov_Matrix(dim)
    st.write("Covariance Matrix:")
    st.table(pd.DataFrame(mat_cov, columns=[f"Dim_{i+1}" for i in range(dim)], index=[f"Dim_{i+1}" for i in range(dim)]))

    # Check if the covariance matrix is positive semi-definite
    if not is_positive_semi_definite(mat_cov):
        st.error("The covariance matrix is not positive semi-definite. Please provide a valid matrix.")
    else:
        st.success("The covariance matrix is positive semi-definite.")

        # Number of samples
        sample = n_sample()
        st.write(f'Number of samples: {sample}')

        # Simulate data
        samples = np.random.multivariate_normal(vec_mu, mat_cov, int(sample))
        
        # Display the simulated data as a table
        st.write("### Simulated Data Table:")
        st.dataframe(pd.DataFrame(samples, columns=[f"Dim_{i+1}" for i in range(dim)]))

    # Visualize for bivariate case (2 dimensions)
    if dim == 2:
        st.write("2D Visualization:")
        X = samples[:, 0]
        Y = samples[:, 1]
        fig, ax = plt.subplots()
        ax.plot(X, Y, 'ro', markersize=3)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('2D Scatter plot of simulated data')
        st.pyplot(fig)

        # Create a meshgrid for plotting the PDF
        X_grid = np.linspace(np.min(X), np.max(X), 100)
        Y_grid = np.linspace(np.min(Y), np.max(Y), 100)
        X_mesh, Y_mesh = np.meshgrid(X_grid, Y_grid)
        pos = np.dstack((X_mesh, Y_mesh))
        pdf = multivariate_normal(vec_mu, mat_cov).pdf(pos)

        # Plot the PDF and contour for bivariate case
        cb_1 = st.checkbox("Show Bivariate normal distribution (3D surface plot)")
        if cb_1:
            my_plot_mvn_3d_surface(X_grid, Y_grid, pdf)

        cb_2 = st.checkbox("Show Contour plot")
        if cb_2:
            my_plot_contour(pdf, X_grid, Y_grid)

    # Display simulated mean vector and covariance matrix
    st.write("### Simulated mean vector (computed from samples):")
    st.table(pd.DataFrame(np.mean(samples, axis=0).reshape(1, -1), columns=[f"Xbar_{i+1}" for i in range(dim)]))

    st.write("### Simulated covariance matrix (computed from samples):")
    st.table(pd.DataFrame(np.cov(samples, rowvar=False), columns=[f"Dim_{i+1}" for i in range(dim)], index=[f"Dim_{i+1}" for i in range(dim)]))

    # Add button to navigate to the analysis page
    if st.button("Go to Analysis Page"):
        st.write("Navigating to the analysis page...")
        st.session_state.page = "Analysis"

