from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
import json, os, time, sys, config, requests
from .services import edgeProcessing
from .services.edgeProcessing import model, exp

api = Blueprint("api", __name__, url_prefix="/api")

@api.route("/edge/modelConfiguration", methods=["POST"])
def edgeModelConfiguration():

	data = request.json
	#model.model_params = data
	#model.load_model()
	#model.load_temperature()
	#model.transform_input_configuration()
	return jsonify({"status": "ok"}), 200
