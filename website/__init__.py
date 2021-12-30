from flask import Flask
from os import path
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'

   
    from .controllers.cvController import cv


    app.register_blueprint(cv, url_prefix='/cv')

    return app