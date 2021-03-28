import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

import CategoryDictionary as CatDic
import FCN_NetModel as FCN

@st.cache(allow_output_mutation=True)
def load_cnn_model(UseGPU=False):
    trained_model_path = 'model/TrainedModelWeiht1m_steps_Semantic_TrainedWithLabPicsAndCOCO_AllSets.torch'
    net = FCN.Net(CatDic.CatNum) 
    if UseGPU==True:
        net.load_state_dict(torch.load(trained_model_path))
    else:
        net.load_state_dict(torch.load(trained_model_path, map_location=torch.device('cpu')))
        
    return net

model = load_cnn_model()

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
    
st.write(
    """ # Detecting, segmenting and classifying materials inside mostly transparent vessels  """
)
prediction_placeholder = st.empty()