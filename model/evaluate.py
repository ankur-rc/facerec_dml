from __future__ import division
from multiprocessing import Process, Queue
import numpy as np
import tqdm

from Dataset import Dataset
from FaceRecognizer import FaceRecognizer


def predict_one(idx, recognizer_obj, image, queue):

    queue.put((idx, recognizer_obj.model.predict(image)))


if __name__ == "__main__":

    dataset = Dataset(
        "/media/ankurrc/new_volume/softura/facerec/datasets/norm_standard_att")

    recognizer_model = FaceRecognizer()

    output = Queue()

    X_train, y_train = dataset.load_data(is_train=True, fold=1, num_train=2)
    print "Training recognizer (", len(X_train), "samples and", len(
        np.unique(y_train)), "subjects)..."
    recognizer_model.train(X_train, y_train)
    print "completed."

    X_test, y_test = dataset.load_data(is_train=False, fold=1, num_train=2)

    print "Predicting on (", len(X_test), "samples)..."

    # pbar = tqdm.tqdm(total=len(X_test))

    # predictions = recognizer_model.predict(X_test)
    processes = [Process(target=predict_one,
                         args=(idx,
                               recognizer_model,
                               image,
                               output))
                 for idx, image in enumerate(X_test)]

    for p in processes:
        p.start()

    predictions = [output.get() for p in processes]

    for p in processes:
        p.join()

    print "Done"

    predictions = [(x[1][0], x[1][1])
                   for x in sorted(predictions, key=lambda x: x[0])]

    recognizer_model.evaluate(predictions, y_test)
