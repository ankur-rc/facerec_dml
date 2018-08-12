from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import os
import tqdm
import numpy as np
import dlib
import cv2
import sklearn

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

"""
Deep Learning based face recognition module.
"""


class FaceRecognizer():

    """
    Trains a face recognition model based on deep metric learning.
    https://arxiv.org/abs/1503.03832
    """

    def __init__(self, model_path="dlib_face_recognition_resnet_model_v1.dat", shape_predictor_path=None, svm_model_path=None):
        """
        Instantiate a FaceRecognizer object
        """

        self.model = dlib.face_recognition_model_v1(model_path)
        if shape_predictor_path is not None:
            self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
        else:
            self.shape_predictor = None

        if svm_model_path is not None:
            self.load(svm_model_path)

        self.logger = logging.getLogger(__name__)

    def train(self):
        """
        Train the recognizer on the training set. Not required when working with pre-trained models.
        TODO: Implement the training mechanism
        """
        raise NotImplementedError()

    def embed(self, images, landmarks=None):
        """
        Generates embeddings for the given images

        :param images: the images to get embeddings of
        :type images: list of numpy.nadarray: (num_images, image_height, image_width)
        :param landmarks: the facial landmarks of the images
        :type landmarks: list, shape: (num_images, 5, 2)
        :return embeddings: the face embeddings
        :rtype predictions: list
        """

        embeddings = []

        if landmarks is None:
            for i in tqdm.tnrange(0, len(images)):
                img = images[i]
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                rect = dlib.rectangle(top=0, left=0, bottom=(
                    (img.shape)[1]-1), right=((img.shape)[0]-1))

                shape = self.shape_predictor(img, rect)

                embedding = self.model.compute_face_descriptor(img, shape)
                embeddings.append(embedding)
                i += 1

        else:
            # convert from gray to rgb
            # images = np.array(images)
            # images = images.reshape(images.shape + (1,))
            # images = np.repeat(images, 3, axis=3)

            images = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in images]

            assert len(images) == len(landmarks)

            embeddings = [self.model.compute_face_descriptor(
                image, landmarks[i]) for i, image in enumerate(images)]

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
        """
        Evaluate the trained SVC on the training set.

        :param X_test: the test embeddings
        :type X_test: numpy.ndarray, shape=(N,128)
        :param ground_truths: the ground truth data corresponding to the test set
        :type ground_truths: numpy.ndarray, shape=(N,1)
        """

        encoder = LabelEncoder()

        X_test = np.array(X_test)
        y_test = encoder.fit_transform(ground_truths)

        acc_svc = accuracy_score(y_test, self.svc.predict(X_test))

        precision_perc = acc_svc*100

        return precision_perc

    def save(self, model_path):
        """
        Save the trained classifier. 
        Call only after fitting the embeddings, otherwise will throw an exception.

        :param model_path: path along with name specifiying where to save the model. Extension should be .pkl for brevity.
        :type model_path: string 
        """
        joblib.dump(self.svc, model_path)

    def load(self, model_path):
        """
        Load the saved classifier model.

        :param model_path: path to the trained classifier model
        :type model_path: string
        """

        self.svc = joblib.load(model_path)

    def infer(self, embeddings, threshold=0.20, unknown_index=-1, in_parallel=False):
        """
        Infer and return a predicted face identity.

        :param embeddings: 128D face embeddings
        :type embeddings: list
        :param threshold: probability threshold to accept a prediction result
        :type threshold: float
        :param unknown_index: a integer id that denotes an unknown class
        :type unknown_index: int
        :param in_parallel: flag to indicate whethetr to run inference in parallel among cpu cores
        :typr in_parallel: bool 
        :return: an identity
        :rtype: int
        """

        unknown = unknown_index

        # get prediction probabilities across all classes for each sample
        predictions = self.svc.predict_proba(np.array(embeddings))

        # get the index of the highest predicted class
        prediction_indices = np.argmax(predictions, axis=1)
        # get the probability of the highest predicted class
        prediction_probabilities = np.max(predictions, axis=1)

        # get the boolean mask for all indices that have a probability less than the threshold value
        thresholded_probabilities = prediction_probabilities < threshold
        # extract the indices from the boolean mask
        thresholded_indices = np.nonzero(thresholded_probabilities)
        # set the indices below the threshold to belong to an unknown class
        prediction_indices[thresholded_indices] = unknown

        # get the index that occured the most in the batch that was evaluated
        predicted_identity = np.max(prediction_indices)

        return predicted_identity
