from flask import Flask, redirect, url_for, render_template, request, flash, session
import re
import os
import sqlite3
import hashlib
from base64 import decode
import requests
import json
from nutritionix import Nutritionix
from decimal import Decimal
from datetime import date, datetime,timedelta 
import time
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder 
import random
from flask import Flask, render_template, request
import os
import base64
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key='honsproject'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.jinja_env.globals.update(zip=zip)

bf_meal = []
onefood = []
lunch_meal = []
dinner_meal = []
snack_meal = []

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

class_mapping = {
    0: {'label': 'aloo-gobi', 'calories': 108},
    1: {'label': 'aloo-fry', 'calories': 125},
    2: {'label': 'dum-aloo', 'calories': 164},
    3: {'label': 'fish-curry', 'calories': 241},
    4: {'label': 'ghevar', 'calories': 61},
    5: {'label': 'green-chutney', 'calories': 21},
    6: {'label': 'gulab-jamun', 'calories': 145},
    7: {'label': 'idli', 'calories': 40},
    8: {'label': 'jalebi', 'calories': 150},
    9: {'label': 'chicken-seekh-kebab', 'calories': 158},
    10: {'label': 'kheer', 'calories': 266},
    11: {'label': 'kulfi', 'calories': 136},
    12: {'label': 'bhature', 'calories': 230}, 
    13: {'label': 'lassi', 'calories': 183},
    14: {'label': 'mutton-curry', 'calories': 298},
    15: {'label': 'onion-pakoda', 'calories': 80},
    16: {'label': 'palak-paneer', 'calories': 338},
    17: {'label': 'poha', 'calories': 270},
    18: {'label': 'rajma-curry', 'calories': 235},
    19: {'label': 'rasmalai', 'calories': 188},
    20: {'label': 'samosa', 'calories': 308},
    21: {'label': 'shahi-paneer', 'calories': 261},
    22: {'label': 'white-rice', 'calories': 135},
    23: {'label': 'bhindi-masala', 'calories': 225},
    24: {'label': 'chicken-biryani', 'calories': 348},
    25: {'label': 'chai', 'calories': 54},
    26: {'label': 'chole', 'calories': 311},
    27: {'label': 'coconut-chutney', 'calories': 105},
    28: {'label': 'dal-tadka', 'calories': 260},
    29: {'label': 'dosa', 'calories': 106}
}

def calculate_total_calories(class_label, count):
    class_info = class_mapping.get(class_label, {'label': 'unknown', 'calories': 0})
    calories_per_item = class_info['calories']
    total_calories = count * calories_per_item
    return total_calories

def detect_and_visualize(img, model_path, class_mapping, confidence_threshold=0.25):
    model = YOLO(model_path)

    results = model.predict(source=img, conf=confidence_threshold)
    detected_items = [0]*30
    float_detections = results[0].boxes.xyxy.tolist()
    detections = [[int(value) for value in detection] for detection in float_detections]
    confidences = results[0].boxes.conf.tolist()
    float_classes = results[0].boxes.cls.tolist()
    classes = [int(value) for value in float_classes]

    total_calories = 0
    resized_img = cv2.resize(img, (800, 400))

    scaling_factor_x = 800 / img.shape[1]
    scaling_factor_y = 400 / img.shape[0]

    for i in range(len(detections)):
        box = detections[i]
        resized_box = [
            int(box[0] * scaling_factor_x),
            int(box[1] * scaling_factor_y),
            int(box[2] * scaling_factor_x),
            int(box[3] * scaling_factor_y)
        ]
        class_index = classes[i]
        class_info = class_mapping.get(class_index, {'label': 'unknown', 'calories': 0})
        conf = confidences[i]
        if conf > 0.4:
            detected_items[class_index] += 1

            class_label = class_info['label']
            calories = class_info['calories']
            total_calories += calories

            count = 1
            total_calories += calculate_total_calories(class_label, count)

            cv2.putText(resized_img, f'{class_label} ({calories} kcal) {conf:.3f}', (resized_box[0], resized_box[1]), cv2.FONT_HERSHEY_PLAIN, 1,(255, 0, 0), 2)
            cv2.rectangle(resized_img, (resized_box[0], resized_box[1]), (resized_box[2], resized_box[3]), (255, 0, 255), 2)
    
    # cv2.putText(resized_img, f'Total Calories: {total_calories:.2f} cal', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    # Convert the OpenCV image to bytes
    _, result_image = cv2.imencode('.jpg', resized_img)
    result_bytes = result_image.tobytes()

    items_with_calories = []
    for i in range(30):
        if(detected_items[i] != 0):
            item_cal = class_mapping[i].get('calories') * detected_items[i]
            items_with_calories.append({'label': class_mapping[i].get('label'), 'calories': f"{detected_items[i]} * {class_mapping[i].get('calories')}.00 = {item_cal}", 'count': detected_items[i]})
    return result_bytes, total_calories, items_with_calories
