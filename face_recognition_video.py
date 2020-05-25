#!/usr/bin/env python
# coding: utf-8

# In[1]:


from imutils.video import VideoStream
import face_recognition
import imutils
import pickle
import time
import cv2
import numpy

encodes_dict = pickle.loads(open(r"C:\Users\krish\Google Drive\facerecog\face_encodes_colab.pkl","rb").read())

vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb_frame,model="cnn")
    test_encodings = face_recognition.face_encodings(rgb_frame, locs)
    
    face_tags =[]
    for encoding in test_encodings:
        face_match=face_recognition.compare_faces(encodes_dict["encodes"],encoding)
        tag = "<Unknown>"
        if True in face_match:
            matched_idxs = numpy.where(face_match)[0]
            list_tags = [encodes_dict["tags"][i] for i in matched_idxs]
            tag = max(set(list_tags), key = list_tags.count)
        face_tags.append(tag)

    for ((t, r, b, l), tag) in zip(locs, face_tags):
        cv2.rectangle(frame, (l, t), (r, b), (90,85,68), 2)
        cv2.rectangle(frame, (l,b),(r,b+40),(90,85,68),-1)
        cv2.putText(frame,tag,(l+10,b+30),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

