from flask import Flask, render_template, Response
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np

app = Flask(__name__)


# hand detector
detector = HandDetector(maxHands=1)
classifier = Classifier("models\model2\keras_model.h5" , "models\model2\labels.txt")
# constants
offset = 20
labels = ["A" , "B" , "C" , "Yes" , "No"]
font = cv2.FONT_HERSHEY_SIMPLEX

def gen_frames():
    camera = cv2.VideoCapture(0) 
    while(camera.isOpened()):
      
    # Capture the video frame
    # by frame
        ret, frame = camera.read()
        
        hands , img = detector.findHands(frame)
        # img = cv2.resize(img , (700 , 700))
        image_white = np.ones((300 , 300 , 3) , dtype=np.uint8) * 255


        if hands:
            hand = hands[0]
            x , y , w , h = hand['bbox']
            img_crop = img[y-offset:y+h+offset , x-offset:x+w+offset]
            

            img_crop_shape = img_crop.shape
            
            aspect_ratio = h / w
            if aspect_ratio > 1:
                new_w = w * (img_crop_shape[0] // h) 
                try:
                    img_resize = cv2.resize(img_crop , (new_w , 300))
                except:
                    continue
                wGap = (300 - new_w) // 2

                image_white[0:img_resize.shape[0] , wGap:new_w+wGap] = img_resize
                prediction , index = classifier.getPrediction(image_white)
                cv2.putText(img, labels[index], (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
                #print(labels[index])

            else:
                new_h = h * (img_crop_shape[1] // w) 
                try:
                    img_resize = cv2.resize(img_crop , (300 , new_h))
                except:
                    continue
                hGap = (300 - new_h) // 2

                image_white[hGap:new_h+hGap , 0:img_resize.shape[1]] = img_resize
                prediction , index = classifier.getPrediction(image_white)
                cv2.putText(img, labels[index], (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
                #print(labels[index])
            # cv2.imshow('img_crop', img_crop)
            # cv2.imshow("Image_white" , image_white)

        
        # Display the resulting frame
        ret, buffer = cv2.imencode('.jpg', img)
        byte_frame = buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + byte_frame + b'\r\n')  # concat frame one by one and show result
    camera.release()
        
@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera')
def Camera():
    return render_template('camera.html')

@app.route('/Learn')
def Learn():
    return render_template('learn.html')

@app.route('/Feedback')
def Feedback():
    return render_template('feedback.html')

@app.route('/Contact')
def Contact():
    return render_template('contact.html')

@app.route('/About')
def About():
    return render_template('about.html')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)