# import the opencv library
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
  

# constants
offset = 20
labels = ["A" , "B" , "C" , "Yes" , "No"]
font = cv2.FONT_HERSHEY_SIMPLEX

# define a video capture object
vid = cv2.VideoCapture(0)

# hand detector
detector = HandDetector(maxHands=1)
classifier = Classifier("models\model2\keras_model.h5" , "models\model3\labels.txt")

  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    
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
            img_resize = cv2.resize(img_crop , (new_w , 300))

            wGap = (300 - new_w) // 2

            image_white[0:img_resize.shape[0] , wGap:new_w+wGap] = img_resize
            prediction , index = classifier.getPrediction(image_white)
            cv2.putText(img, labels[index], (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
            print(labels[index])

        else:
            new_h = h * (img_crop_shape[1] // w) 
            img_resize = cv2.resize(img_crop , (300 , new_h))

            hGap = (300 - new_h) // 2

            image_white[hGap:new_h+hGap , 0:img_resize.shape[1]] = img_resize
            prediction , index = classifier.getPrediction(image_white)
            cv2.putText(img, labels[index], (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
            print(labels[index])
        # cv2.imshow('img_crop', img_crop)
        # cv2.imshow("Image_white" , image_white)

    
    # Display the resulting frame
    cv2.imshow('Image', img)
    

    # the 'q' button is set as the quitting button you may use any desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()