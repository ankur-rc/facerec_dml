from __future__ import division
import os
import cv2
import pickle
import numpy as np
import tqdm
from multiprocessing import Process, Queue

from Dataset import Dataset

"""
Trains a face recognition model based on the given dataset and algorithm
"""


class FaceRecognizer():

    def __init__(self):
        """
        Instantiate a FaceRecognizer object
        """

        self.model = cv2.face.LBPHFaceRecognizer_create()

    def train(self, images, labels):
        """
        Train the recognizer on the training set

        :param images: the images to train on
        :type images: numpy.ndarray shape: (num_images, image_height, image_width)
        :param labels: the labels/subjects the corresponding faces belong to
        :type labels: numpy.ndarray shape: (num_images,)
        """

        self.model.train(images, labels)

    def predict(self, images):
        """
        Predicts the labels of the given images

        :param images: the images to test on
        :type images: numpy.ndarray shape: (num_images, image_height, image_width)
        :return predictions: the predicted labels
        :rtype predictions: array
        """

        predictions = []
        # for i in tqdm.trange(0, len(images)):
        #     prediction = preds_with_indx(i, images[i])
        #     predictions.append(prediction)
        #     i += 1

        print predictions
        return predictions

    def evaluate(self, predictions, ground_truths):

        assert(len(predictions) == len(ground_truths))

        true_positive = 0
        predictions = sorted(predictions, key=lambda x: x[0])

        for idx, pred in enumerate(predictions):
            if pred[1][0] == ground_truths[idx]:
                true_positive += 1

        # true_positive = np.count_nonzero(
        #     np.equal(ground_truths, np.array(predictions)[:, 0]))

        precision_perc = true_positive/len(predictions)*100

        print "Precision@1:", true_positive, "/", len(
            predictions), "(", precision_perc, "%)"

    def save(self, name):

        self.model.write(name)

    def load(self, name):

        self.model.read(name)


def predict_one(idx, recognizer_obj, image, queue):

    queue.put((idx, recognizer_obj.model.predict(image)))


if __name__ == "__main__":

    dataset = Dataset(
        "/media/ankurrc/new_volume/softura/facerec/datasets/norm_cyber_extruder_ultimate")

    recognizer_model = FaceRecognizer()

    output = Queue()

    X_train, y_train = dataset.load_data(is_train=True, fold=1)
    print "Training recognizer (", len(X_train), "samples and", len(
        np.unique(y_train)), "subjects)..."
    recognizer_model.train(X_train, y_train)
    print "completed."

    model_name = os.path.basename(dataset.dataset_path) + "__model.yml"
    print "Saving recognizer model..."
    recognizer_model.save(model_name)
    print "done."

    print "Loading recognizer model..."
    recognizer_model.load(model_name)
    print "done."
    X_test, y_test = dataset.load_data(is_train=False, fold=1)

    print "Predicting on (", len(X_test), "samples)..."
    # predictions = recognizer_model.predict(X_test)
    processes = [Process(target=predict_one, args=(idx,
                                                   recognizer_model, image, output)) for idx, image in enumerate(X_test)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    predictions = [output.get() for p in processes]
    print "Done"

    recognizer_model.evaluate(predictions, y_test)
