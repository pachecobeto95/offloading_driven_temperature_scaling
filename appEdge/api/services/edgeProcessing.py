from flask import jsonify, session, current_app as app
import os, pickle, requests, sys, config, time, math, json, torch, io, datetime
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
#from .utils import ModelLoad, transform_image, ExpLoad
import pandas as pd

def edgeNoCalibInference(fileImg, data_dict):
	return {"status": "ok"}
