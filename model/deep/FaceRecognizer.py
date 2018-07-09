from __future__ import division
import os
import tqdm
import numpy as np
import dlib
import imutils
import cv2

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
            print img.shape
            rect = dlib.rectangle(top=0, left=0, bottom=(
                (img.shape)[1]-1), right=((img.shape)[0]-1))

            shape = self.shape_predictor(img, rect)

            embedding = self.model.compute_face_descriptor(img, shape)
            embeddings.append(embedding)
            i += 1

        return embeddings

    def evaluate(self, predictions, ground_truths):

        assert(len(predictions) == len(ground_truths))

        true_positive = np.count_nonzero(
            np.equal(ground_truths, np.array(predictions)[:, 0]))

        precision_perc = true_positive/len(predictions)*100

        print "Precision@1:", true_positive, "/", len(
            predictions), "(", precision_perc, "%)"

    def save(self, name):

        self.model.write(name)

    def load(self, name):

        self.model.read(name)


if __name__ == "__main__":

    recognizer = FaceRecognizer()
    recognizer.train()
