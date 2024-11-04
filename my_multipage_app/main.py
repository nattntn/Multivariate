import streamlit as st
from streamlit_option_menu import option_menu
import Home
import Upload
import Simulate
import Analysis

# ตรวจสอบการมีอยู่ของ 'page' ใน session_state ถ้าไม่มีให้กำหนดค่าเริ่มต้น
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# สร้าง Sidebar สำหรับการนำทาง
with st.sidebar:
    page = option_menu("Main Menu", ["Home", "Upload File", "Simulate Data", "Analysis"], 
        icons=['house', 'cloud-upload', 'play', 'chart-line'], 
        menu_icon="bars",  
        default_index=["Home", "Upload File", "Simulate Data", "Analysis"].index(st.session_state.page)  # กำหนดให้แสดงหน้าปัจจุบัน
    )

# อัปเดต session_state เมื่อมีการเปลี่ยนแปลงใน option_menu
if page != st.session_state.page:
    st.session_state.page = page

# Navigate to the selected page
if st.session_state.page == "Home":
    Home.app()
elif st.session_state.page == "Upload File":
    Upload.app()
elif st.session_state.page == "Simulate Data":
    Simulate.app()
elif st.session_state.page == "Analysis":
    Analysis.app()
