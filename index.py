import cv2 as cv
import os
import numpy as np
import time
import face_recognition
import imutils
import pickle
from threading import Thread

faceCascade = cv.CascadeClassifier('face_cascade.xml')
data = pickle.loads(open('face_enc', "rb").read())

ip = "10.34.113.119"
port = "8080"

class VideoStreamWidget(object):
    def __init__(self):

        addr = 'http://' + ip +":"+port+"/video"
        self.capture = cv.VideoCapture(addr)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):

        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(.01)

    def show_frame(self):

        gray = cv.cvtColor(self.frame,cv.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(60, 60),
                                     flags=cv.CASCADE_SCALE_IMAGE)

        rgb = cv.cvtColor(gray, cv.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        names = []

        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"
            if True in matches:

                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                #nu stiu ce dracu fac dar simt ca viata mea atarna de un fir de ata, informatica
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                name = max(counts, key=counts.get)
            names.append(name)
        cv.imshow("Frame", self.frame)


if __name__ == '__main__':
    video_stream_widget = VideoStreamWidget()
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError:
            pass
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
