import socketio
import eventlet
import numpy as np
import cv2
import base64
from flask import Flask
from keras.models import load_model
from io import BytesIO
from PIL import Image


#########################
#  Preprocessing images #
#########################
def img_preprocess(img):
  # Step 1 : Crop unnecesary noise
  img = img[60:135, :,:]
  # Step 2 : Change color space of the image
  # * The NVIDIA architecture recommend recommend using YUV color space
  # as opposed to RGB used in the leNet architecture
  # * Y : Uses the luminosity or the brightness of the image
  # * UV : Represents chromium that add colos to the image
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  # Step 3: Gaussian blur helps smooth the image, reduce noise 
  # by applying gaussian kernel convolutional
  img = cv2.GaussianBlur(img, (3,3), 0)
  # Step 4: Resize so that ita matches the input size of the NVIDIA mode
  img = cv2.resize(img,(200,66))
  # Step 5: Normalizing the image. No visual impact on the image
  img = img/255
    
  return img
 



# Flask: microframework to build web apps
# * __name__ : WWhenever you execture python script, python assigns the name main
#######################
# 1. Example of Flask #
#######################

# * Tells Flask what url we should use to trigger
# greeting function
## CODE
# app = Flask(__name__ ) #'__main__'
# @app.route('/hpome')
#def greeting():
#   return "Welcome!"

#if __name__ == '__main__':
    # if we execute the scipt we want to run the application
    #app.run(port = 3000)

#######################
# 2.  Flask & Socket. #
#######################
# * We want to create bidirectional communication between our model
# and the simulator. For the simulator
# * Server() : Real time communication between client/server
# * Flask app : Middleware to dispatch traffic to a socket server
app = Flask(__name__)
sio = socketio.Server()
speed_limit = 10

# to fire the respectice functions



@sio.on('telemetry')
def telemetry(sid, data):
    # * Data from the simulator. In case of an event the simulator 
    # is fired with the relevant data
    # * the model will extract the features from the image and will
    # predict the steering angle which we send back
    # * image: 64 decoded
    # * use also of a buffer module to mimic our data like a normal file
    # which we can further use for processing

    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))

    # Step 2: It must be preprocessed the same way!
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])

    # Step 3: Predict the steering angle
    steering_angle = float(model.predict(image))
    # enforce constant speed
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)
 

@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)
 
def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })
 
 
if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    # * WSG I : The server sends any requests made by the client to
    # the web app itself.
    # * listen( ar1 = ('':for any available IP address, 4567: port)) 
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)