
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


from models import ResNetMNIST, CustomMNIST

import numpy as np
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import torch
import pytorch_lightning as pl
from torchvision.models import resnet18
from torch import nn
from torchvision import datasets
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


@st.cache(suppress_st_warning=True)
def get_pretrained_model(model_name):
    st.info('Actually loading the pretrained model')
    if model_name == 'resnet':
        inference_model = ResNetMNIST.load_from_checkpoint("./"+model_name+"_mnist.pt")
    elif model_name == 'custom':
        inference_model = CustomMNIST.load_from_checkpoint("./"+model_name+"_mnist.pt")
    else:
        st.error("model name doesn't exist")
        st.stop()
    inference_model.eval()
    
    return inference_model
    

st.title('MNIST example')

st.markdown('''
Draw a number from 0 to 9
''')


SIZE = 280
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=10,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw",
    key='canvas')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    
model_option = st.selectbox(
     'Select the architecture',
     ('resnet', 'custom'))

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test_x = test_x.reshape(1, 28, 28)
    test_x = transform(test_x)
    test_x = test_x.permute(1, 2, 0)
    
    inference_model = get_pretrained_model(model_option)
    if model_option == 'resnet':
        probabilities = torch.softmax(inference_model(test_x.unsqueeze(1)), dim=1)
    else:
        probabilities = torch.softmax(inference_model(torch.flatten(test_x, start_dim=1, end_dim=2).unsqueeze(1)[0]), dim=1)
        
    predicted_class = torch.argmax(probabilities, dim=1)
    st.info('Predicted class: '+str(predicted_class.item()))
    st.bar_chart(probabilities.cpu().detach().numpy().flatten())