import cv2
import numpy as np
import streamlit as st
import torch

import CategoryDictionary as CatDic
import FCN_NetModel as FCN

@st.cache(allow_output_mutation=True)
def load_cnn_model(use_gpu_flag=False):
    trained_model_path = 'model/TrainedModelWeiht1m_steps_Semantic_TrainedWithLabPicsAndCOCO_AllSets.torch'
    net = FCN.Net(CatDic.CatNum) 
    if use_gpu_flag==True:
        net.load_state_dict(torch.load(trained_model_path))
    else:
        net.load_state_dict(torch.load(trained_model_path, map_location=torch.device('cpu')))
        
    return net

def do_predictions(input_image, use_gpu_flag, freeze_batch_norm_statistics_flag, model):
    h,w,d=input_image.shape
    r=np.max([h,w])
    if r>840:
        fr=840/r
        input_image=cv2.resize(input_image,(int(w*fr),int(h*fr)))
    img_to_array=np.expand_dims(input_image, axis=0)
    with torch.autograd.no_grad():
          out_prob_dict, out_lb_dict=model.forward(Images=img_to_array, TrainMode=False, UseGPU=use_gpu_flag, FreezeBatchNormStatistics=freeze_batch_norm_statistics_flag)

    return out_prob_dict, out_lb_dict, input_image

def plot_predictions(out_lb_dict, resized_image):
    for category_name in out_lb_dict:
        lb=out_lb_dict[category_name].data.cpu().numpy()[0].astype(np.uint8)
        if lb.mean()<0.001: continue
        if category_name=='Ignore': continue
        im_overlay = resized_image.copy()
        im_overlay[:, :, 0][lb==1] = 255
        im_overlay[:, :, 1][lb==1] = 0
        im_overlay[:, :, 2][lb==1] = 255
        final_image=np.concatenate([resized_image, im_overlay], axis=1)
        st.write(category_name)
        st.image(final_image)

st.write(
    """ # Detecting, segmenting and classifying materials inside mostly transparent vessels  """
)

st.sidebar.info("Inference settings")
use_gpu = st.sidebar.checkbox('Use GPU')
model = load_cnn_model(use_gpu)
freeze_batch_norm_statistics = st.sidebar.checkbox('Freeze Batch Norm Statistics')
st.sidebar.info("Input")
uploaded_file = st.sidebar.file_uploader("Upload a JPG image of vessel(s)", type=['jpg'])
if uploaded_file is not None:
    st.sidebar.text('Input image:')
    st.sidebar.image(uploaded_file, width=288)
    predict_button = st.sidebar.button('Predict')
    if predict_button:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        out_prob_dict, out_lb_dict, resized_image = do_predictions(image, use_gpu, freeze_batch_norm_statistics, model)
        plot_predictions(out_lb_dict, resized_image)