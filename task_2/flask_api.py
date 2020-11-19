# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 17:16:32 2020

@author: elton
"""

import flask
from flask import Flask, request
import flasgger
from flasgger import Swagger
import pandas as pd
import numpy as np
import pickle

flask_api=Flask(__name__)
Swagger(flask_api)

pickle_in = open('regressor1.pkl', 'rb')
regressor1=pickle.load(pickle_in)

@flask_api.route('/') #root 
def welcome():
    return "Welcome All"

@flask_api.route('/predict', methods=["GET"]) #get method
def predict_compressor_decay_state_coefficient():
    
    """predict compressor decay state coefficient
    ---
    parameters:
      - name: Lever position
        in: query
        type: number
        required: true
      - name: Ship speed
        in: query
        type: number
        required: true  
      - name: Gas turbine shaft torque
        in: query
        type: number
        required: true  
      - name: Gas turbine rate of revolutions
        in: query
        type: number
        required: true
      - name: Gas generator rate of revolutions
        in: query
        type: number
        required: true
      - name: Starboard propeller torque
        in: query
        type: number
        required: true  
      - name: Port propeller torque
        in: query
        type: number
        required: true  
      - name: HP turbine exit temperature
        in: query
        type: number
        required: true  
      - name: GT compressor outlet air temperature
        in: query
        type: number
        required: true
      - name: HP turbine exit pressure
        in: query
        type: number
        required: true
      - name: GT compressor outlet air pressure
        in: query
        type: number
        required: true
      - name: Gas turbine exhaust gas pressure
        in: query
        type: number
        required: true
      - name: Turbine injection control
        in: query
        type: number
        required: true  
      - name: Fuel flow
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
            
            
    """
            
    lever_position=request.args.get('Lever position')
    ship_speed=request.args.get('Ship speed')
    gt_shaft=request.args.get('Gas turbine shaft torque')
    gt_rate=request.args.get('Gas turbine rate of revolutions')
    gg_rate=request.args.get('Gas generator rate of revolutions')
    sp_torque=request.args.get('Starboard propeller torque')
    pp_torque=request.args.get('Port propeller torque')
    hpt_temp=request.args.get('HP turbine exit temperature')
    gt_c_o_temp=request.args.get('GT compressor outlet air temperature')
    hpt_pressure=request.args.get('HP turbine exit pressure')
    gt_c_o_pressure=request.args.get('GT compressor outlet air pressure')
    gt_exhaust_pressure=request.args.get('Gas turbine exhaust gas pressure')
    turbine_inj_control=request.args.get('Turbine injection control')
    fuel_flow=request.args.get('Fuel flow')
    prediction = regressor1.predict([[lever_position, ship_speed, gt_shaft,
                                      gt_rate, gg_rate, sp_torque, pp_torque,
                                      hpt_temp, gt_c_o_temp, hpt_pressure,
                                      gt_c_o_pressure, gt_exhaust_pressure,
                                      turbine_inj_control, fuel_flow]])
    return "The predicted value is " + str(prediction)

@flask_api.route('/predict_file', methods=["POST"])
def predict_compressor_decay_state_coefficient_file():
    """predict compressor decay state coefficient
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        
    responses:
        200:
            description: The output values
            
    """
    
    test_data=pd.read_csv(request.files.get("file"))
    test_data.drop('Unnamed: 0', axis=1, inplace=True)
    test_data.columns = ['lever_position', 'ship_speed', 'gt_shaft', 'gt_rate',
                     'gg_rate', 'sp_torque', 'pp_torque', 'hpt_temp',
                     'gt_c_i_temp', 'gt_c_o_temp', 'hpt_pressure',
                     'gt_c_i_pressure', 'gt_c_o_pressure',
                     'gt_exhaust_pressure', 'turbine_inj_control', 'fuel_flow',
                     'gt_c_decay',  'gt_t_decay']
    test_data.drop(['gt_c_i_temp', 'gt_c_i_pressure', 'gt_c_decay', 'gt_t_decay'],
          axis=1 ,inplace=True)
    
    prediction=regressor1.predict(test_data)
    return "The predicted value is " + str(list(prediction))
    
if __name__ == '__main__':
    flask_api.run()