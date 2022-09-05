from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
import json, os, time, sys, config, requests
from .services import cloudProcessing
from .services.cloudProcessing import model, exp


api = Blueprint("api", __name__, url_prefix="/api")


@api.route("/cloud/modelConfiguration", methods=["POST"])
def cloudModelConfiguration():
	data = request.json
	#model.model_params = data
	#model.load_model()
	#model.load_temperature()
	return jsonify({"status": "ok"}), 200
