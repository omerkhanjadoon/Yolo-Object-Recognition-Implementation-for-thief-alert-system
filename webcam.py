
import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import winsound

#to save video
#python flow --model cfg/yolo.cfg --load bin/yolov2.weights --demo videofile.avi --saveVideo


options = {
    'model': 'cfg/yolo.cfg',
    'load':'bin/yolov2.weights',
    'threshold': 0.15
}

tfnet = TFNet(options)


capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


codec = cv2.VideoWriter_fourcc(*'XVID') 
output1 = cv2.VideoWriter('output1.avi', codec, 60.0,(1920,1080))
colors = [tuple(255 * np.random.rand(3)) for _ in range(5)]

i = 0
frame_rate_divider = 3
while(capture.isOpened()) :
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        if i % frame_rate_divider == 0:
            results = tfnet.return_predict(frame)
        
            for color, result in zip(colors, results):
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                label = result['label']
                confidence = result['confidence']
                confidence_percent = confidence *100 #confidence percentage
                if label=="person" and confidence_percent>45:
                	winsound.PlaySound("siren.wav",winsound.SND_ASYNC)
                
                text = '{}: {:.0f}%'.format(label, confidence * 100)
                frame = cv2.rectangle(frame, tl, br, color, 5)
                frame = cv2.putText(frame, text, tl,cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            output1.write(frame)
            cv2.imshow('frame', frame)
            print('FPS {:.1f}'.format(1 / (time.time() - stime)))
            i +=1 
        else:
            i +=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
        
capture.release()
output1.release()
cv2.destroyAllWindows()