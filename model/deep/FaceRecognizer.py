from __future__ import division
import os
import tqdm
import numpy as np
import dlib
import imutils
import cv2

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

"""
Trains a face recognition model based on deep metric learning for the given dataset and algorithm
"""


class FaceRecognizer():

    def __init__(self, model_path="dlib_face_recognition_resnet_model_v1.dat", shape_predictor_path="shape_predictor_5_face_landmarks.dat"):
        """
        Instantiate a FaceRecognizer object
        """

        self.model = dlib.face_recognition_model_v1(model_path)
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def train(self):
        """
        Train the recognizer on the training set. Not required when working with pre-trained models.
        TODO: Implement the training mechanism
        """
        raise NotImplementedError()

    def embed(self, images):
        """
        Generates embeddings for the given images

        :param images: the images to test on
        :type images: numpy.ndarray shape: (num_images, image_height, image_width)
        :return embeddings: the face embeddings
        :rtype predictions: array
        """

        embeddings = []
        for i in tqdm.trange(0, len(images)):
            img = images[i]
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            rect = dlib.rectangle(top=0, left=0, bottom=(
                (img.shape)[1]-1), right=((img.shape)[0]-1))

            shape = self.shape_predictor(img, rect)

            embedding = self.model.compute_face_descriptor(img, shape)
            embeddings.append(embedding)
            i += 1

        return embeddings

    def fit_embeddings(self, embeddings, labels):
        """
        Trains a linear SVC based on the embeddings and labels.

        :param embeddings: list of 128-D lists, each representaing a face embedding
        :type embeddings: list of list
        :param labels: array of labels corresponding to each embeddings
        :type labels: numpy.array
        """

        self.svc = LinearSVC()

        encoder = LabelEncoder()

        X_train = np.array(embeddings)
        y_train = encoder.fit_transform(labels)

        self.svc.fit(X_train, y_train)

    def evaluate(self, X_test, ground_truths):

        encoder = LabelEncoder()

        X_test = np.array(X_test)
        y_test = encoder.fit_transform(ground_truths)

        acc_svc = accuracy_score(y_test, self.svc.predict(X_test))

        precision_perc = acc_svc*100

        print "Precision@1:", precision_perc, "%"

    def save(self, name):

        self.model.write(name)

    def load(self, name):

        self.model.read(name)


if __name__ == "__main__":

    recognizer = FaceRecognizer()
    recognizer.train()
