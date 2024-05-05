import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

# Set up SocketIO server and Flask app
sio = socketio.Server()
app = Flask(__name__) #'__main__'
speed_limit = 10  # Define the speed limit for the car

# Preprocessing function for images
def img_preprocess(img):
    # Crop the image to focus on the road
    img = img[60:135,:,:]
    # Convert image to YUV color space (used by the NVIDIA model)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # Resize image to match model input size
    img = cv2.resize(img, (200, 66))
    # Normalize pixel values to the range [0, 1]
    img = img/255
    return img

# Event handler for SocketIO connection
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)  # Send initial control signal to stop the car

# Function to send control commands (steering angle and throttle) via SocketIO
def send_control(steering_angle, throttle):
    # Send steering angle and throttle as JSON data
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),  # Convert to string for serialization
        'throttle': throttle.__str__()  # Convert to string for serialization
    })

# Event handler for receiving telemetry data from the client
@sio.on('telemetry')
def telemetry(sid, data):
    # Extract speed and image data from telemetry
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))  # Decode base64 encoded image
    image = np.asarray(image)  # Convert image to numpy array
    image = img_preprocess(image)  # Preprocess image for model input
    image = np.array([image])  # Add batch dimension for model prediction
    # Predict steering angle using the loaded model
    steering_angle = float(model.predict(image))
    # Calculate throttle based on current speed
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))  # Print telemetry data
    # Send control commands to the car
    send_control(steering_angle, throttle)

if __name__ == '__main__':
    # Load pre-trained Keras model for autonomous driving
    model = load_model('model/model.h5')
    # Wrap Flask app with SocketIO middleware
    app = socketio.Middleware(sio, app)
    # Start the server to listen for client connections
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
