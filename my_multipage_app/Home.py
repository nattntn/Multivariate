import streamlit as st

# หน้า Home
def app():
    # แสดงชื่อและคำแนะนำตัว
    st.title("Welcome to the Multivariate Analysis Web App")
    st.subheader("An Interactive Tool for Multivariate Data Analysis")
    
    # เพิ่มรูปภาพในหน้า Home (ลิงก์จาก URL หรือไฟล์ในเครื่อง)
    st.image("https://img-cdn.pixlr.com/image-generator/history/65bb506dcb310754719cf81f/ede935de-1138-4f66-8ed7-44bd16efc709/medium.webp", caption="Explore Multivariate Data", use_column_width=True)
    
    # คำบรรยายเกี่ยวกับแอป
    st.write("""
    ## Introduction
    This web application allows users to upload datasets, simulate multivariate normal distributions, and perform various analyses.
    You can navigate to different pages using the menu on the sidebar. Below is a brief overview of what each section offers:
    
    - **Upload File**: Upload your datasets and explore the data.
    - **Simulate Data**: Generate random data following a multivariate normal distribution.
    - **Analysis**: Perform statistical tests and visualizations on your data.
    """)

    # ปุ่มนำทางไปยังแต่ละส่วน (Optional)
    st.write("### Quick Navigation")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Go to Upload"):
            st.session_state.page = "Upload File"  # เปลี่ยนหน้าไปยัง "Upload File"
    
    with col2:
        if st.button("Go to Simulate"):
            st.session_state.page = "Simulate Data"  # เปลี่ยนหน้าไปยัง "Simulate Data"
    
    with col3:
        if st.button("Go to Analysis"):
            st.session_state.page = "Analysis"  # เปลี่ยนหน้าไปยัง "Analysis"
