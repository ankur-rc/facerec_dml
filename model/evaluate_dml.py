from __future__ import division
from multiprocessing import Process, Queue
import numpy as np
import tqdm

from Dataset import Dataset
from deep.FaceRecognizer import FaceRecognizer


def predict_one(idx, recognizer_obj, image, queue):

    queue.put((idx, recognizer_obj.model.predict(image)))


if __name__ == "__main__":

    dataset = Dataset(
        "/media/ankurrc/new_volume/softura/facerec/datasets/norm_standard_att")

    recognizer_model = FaceRecognizer(
        model_path="deep/dlib_face_recognition_resnet_model_v1.dat",
        shape_predictor_path="deep/shape_predictor_5_face_landmarks.dat")

    X_train, y_train = dataset.load_data(is_train=True, fold=1, num_train=2)

    print "Training recognizer (", len(X_train), "samples and", len(
        np.unique(y_train)), "subjects)..."

    print "Step 1. Generating embeddings..."
    embeddings = recognizer_model.embed([X_train[0]])
    print "completed."

    print embeddings, type(embeddings)

    # X_test, y_test = dataset.load_data(is_train=False, fold=1, num_train=2)

    # print "Predicting on (", len(X_test), "samples)..."

    # print "Done"
