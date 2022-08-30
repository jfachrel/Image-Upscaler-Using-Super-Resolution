import streamlit as st
import numpy as np
from PIL import Image

from utils import SuperResolution

# Page settings
st.set_page_config(
    page_title="Super Resolution App",
    layout="wide",
    initial_sidebar_state="expanded"
 )

# Title
st.title('Image Upscaler')

# Upload file
uploaded_file = st.file_uploader(label="Choose a file", type=['jpg', 'jpeg'])

sidebar = st.sidebar

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    res = SuperResolution()
    cropped = res.cropped(image=image)
    resizing_image = res.resizing(image=image)
    FSRCNN_x3 = res.resolution(image=image,model='FSRCNN_x3')
    EDSR_x4 = res.resolution(image=image,model='EDSR_x4')
    ESPCN_x4 = res.resolution(image=image,model='ESPCN_x4')
    LapSRN_x8 = res.resolution(image=image,model='LapSRN_x8')

    col1, col2 = st.columns([0.5, 0.5])

    #Col 1
    with col1:
        st.markdown('<p style="text-align: center;">Original Image</p>', unsafe_allow_html=True)
        st.image(image, width=425)
    
    with col1:
        st.markdown('<p style="text-align: center;">Resizing with OpenCV</p>', unsafe_allow_html=True)
        st.image(resizing_image, width=425)
    
    with col1:
        st.markdown('<p style="text-align: center;">ESPCN_x4</p>', unsafe_allow_html=True)
        st.image(ESPCN_x4, width=425)

    with col1:
        st.markdown('<p style="text-align: center;">LapSRN_x8</p>', unsafe_allow_html=True)
        st.image(LapSRN_x8, width=425)

    #Col 2
    with col2:
        st.markdown('<p style="text-align: center;">Original Image After Cropped</p>', unsafe_allow_html=True)
        st.image(cropped, width=425)

    with col2:
        st.markdown('<p style="text-align: center;">FSRCNN_x3</p>', unsafe_allow_html=True)
        st.image(FSRCNN_x3, width=425)
    
    with col2:
        st.markdown('<p style="text-align: center;">EDSR_x4</p>', unsafe_allow_html=True)
        st.image(EDSR_x4, width=425)