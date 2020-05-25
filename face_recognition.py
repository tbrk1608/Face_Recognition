# -*- coding: utf-8 -*-
#imports
from imutils import paths
import face_recognition
import pickle
import cv2
import os
import matplotlib.pyplot as plt
import numpy

# Mapping encodes - name tags 
ds_path = "/content/drive/My Drive/facerecog/dataset"
imgs = list(paths.list_images(ds_path))
encodes_dict= {"encodes": [], "tags": []}
   
for img in imgs:
    image = cv2.imread(img)
    tag = img.split(os.path.sep)[-1].split(' ')[0]
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb_img,model='cnn')
    encodings = face_recognition.face_encodings(rgb_img, locs)
    try:
        encodes_dict['encodes'].append(encodings[0])
        encodes_dict['tags'].append(tag)
    except IndexError:
        print("{} ::: No Face Found".format(img))

# saving to pickle
outfile = "/content/drive/My Drive/facerecog/face_encodes_colab.pkl"
os.makedirs(os.path.dirname(outfile), exist_ok=True)
with open(outfile, 'wb') as f:
    pickle.dump(encodes_dict, f)

print(len(encodes_dict["encodes"]))
print(len(encodes_dict["tags"]))

pkl_path = "/content/drive/My Drive/facerecog/face_encodes_colab.pkl"
encodes_dict = pickle.loads(open(pkl_path, "rb").read())

image_path = "/content/drive/My Drive/facerecog/test/test (4).jpg"
image = cv2.imread(image_path)
rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
locs = face_recognition.face_locations(rgb_img,model='cnn')
test_encodings = face_recognition.face_encodings(rgb_img, locs)

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
    cv2.rectangle(image, (l, t), (r, b), (90,85,68), 2)
    cv2.rectangle(image, (l,b),(r,b+40),(90,85,68),-1)
    cv2.putText(image,tag,(l+10,b+30),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),1)

plt.figure(figsize=(12,8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()