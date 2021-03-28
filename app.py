import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.sidebar.info("Inference settings")
use_gpu = st.sidebar.checkbox('Use GPU')
freeze_batch_norm_statistics = st.sidebar.checkbox('Freeze Batch Norm Statistics')
st.sidebar.info("Input")
uploaded_file = st.sidebar.file_uploader("Upload a JPG or PNG image of vessel(s)", type=['jpg', 'png'])
if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    st.sidebar.text('Input image:')
    st.sidebar.image(uploaded_image, width=288)
    predict_button = st.sidebar.button('Predict')
    clean_button = st.sidebar.button('Clean')
    
st.write(
    """ # Detecting, segmenting and classifying materials inside mostly transparent vessels  """
)
prediction_placeholder = st.empty()