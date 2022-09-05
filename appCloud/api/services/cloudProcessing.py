from flask import jsonify, session, current_app as app
import os, pickle, requests, sys, config, time, json, torch, io, datetime
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
#from .utils import load_model
#from .utils import ModelLoad, transform_image, ExpLoad
import torchvision.models as models


def onlyCloudProcessing(fileImg):
	#try:
	#image_bytes = fileImg.read()
	#response_request = {"status": "ok"}

	#Starts measuring the inference time
	#tensor_img = transform_image(image_bytes) #transform input data, which means resize the input image

	#Run DNN inference
	#output = only_cloud_dnn_inference_cloud(tensor_img)
	return {"status": "ok"}

def cloudNoCalibInference(feature, conf_list, class_list):
	return {"status": "ok"}
