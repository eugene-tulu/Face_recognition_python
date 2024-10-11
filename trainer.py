import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "dataset"

def get_images_with_id(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    for path in image_paths:
        faceImg = Image.open(path).convert('L')
        faceNp = np.array(faceImg, np.uint8)
        id = int(os.path.split(path)[-1].split(".")[1])
        print(id)
        faces.append(faceNp)
        ids.append(id)
        cv2.imshow("Training", faceNp)
        cv2.waitKey(10)
    
    return np.array(ids), faces

ids, faces = get_images_with_id(path)
recognizer.train(faces, ids)
recognizer.save("recognizer/training_data.yml")
cv2.destroyAllWindows()
