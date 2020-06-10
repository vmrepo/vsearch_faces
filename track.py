
import cv2
import numpy as np

CASCADE = cv2.CascadeClassifier('./face_detectors/haarcascade_frontalface_default.xml')
#NET = cv2.dnn.readNetFromCaffe('./face_detectors/deploy.prototxt', './face_detectors/res10_300x300_ssd_iter_140000_fp16.caffemodel')
#NET = cv2.dnn.readNetFromTensorflow('./face_detectors/opencv_face_detector_uint8.pb', './face_detectors/opencv_face_detector.pbtxt')

def netdetect(img):

    detect_confidence_threshold = 0.7

    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))

    NET.setInput(blob)
    detections = NET.forward()
    rects = []

    for i in range(0, detections.shape[2]):
        # filter out weak detections by ensuring the predicted
        # probability is greater than a minimum threshold
        if detections[0, 0, i, 2] > detect_confidence_threshold:
            # compute the (x, y)-coordinates of the bounding box for
            # the object, then update the bounding box rectangles list
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            rects.append((startX, startY, endX - startX, endY - startY))

    return rects

cap = cv2.VideoCapture('./cuts/000/000_001.avi')

while(1):

    success, frame = cap.read()

    if not success:
        break

    rects = CASCADE.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor = 1.2, minNeighbors = 5);
    #rects = netdetect(frame)

    for x,y,w,h in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
