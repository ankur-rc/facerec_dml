import os
import cv2

if __name__ == "__main__":
    recognizer_model = cv2.face.LBPHFaceRecognizer_create()
    recognizer_model.train()
