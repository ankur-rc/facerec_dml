import os
import random
import shutil
import tqdm as pbar
import numpy as np
import cv2


"""
A class for performing various operations on a dataset
"""


class Dataset():

    def __init__(self, dataset_path):
        """
        Instantiate a Dataset object

        :param dataset_path: path to the dataset
        :type dataset_path: str
        """

        if not os.path.exists(os.path.realpath(dataset_path)):
            raise Exception("Invalid dataset path!")

        dataset_split_path_suffix = "__train_test_split"

        self.dataset_path = os.path.realpath(dataset_path)
        self.dataset_split = os.path.basename(
            dataset_path)+dataset_split_path_suffix

    def clear_splits(self):
        if os.path.exists(self.dataset_split):
            print "Removing stale directory:", self.dataset_split, "..."
            shutil.rmtree(self.dataset_split)
            print "done."

    def split(self, num_train=2, fold=1):
        """
        Generates a train-test split based on the number of training samples

        :param num_train: number of training samples per split
        :type num_train: int
        :param fold: the fold the split belongs to
        :type fold: int
        """

        # if os.path.exists(os.path.join(self.dataset_split, str(num_train))):
        #     # print "Removing stale directory:", self.dataset_split, "..."
        #     shutil.rmtree(os.path.join(self.dataset_split, str(num_train)))
        #     # print "done."

        # print "Creating directory:", os.path.join(
        #     self.dataset_split, str(num_train)), "..."
        # os.makedirs(os.path.join(self.dataset_split, str(num_train)))
        # print "done."

        train_file = "train.csv"
        test_file = "test.csv"

        fold_path = self.dataset_split + os.sep + \
            str(num_train) + os.path.sep + str(fold)
        if os.path.exists(fold_path):
            # print "Removing stale directory:", fold_path, "..."
            shutil.rmtree(fold_path)
            # print "done."

        # print "Creating directory:", fold_path, "..."
        os.makedirs(fold_path)
        # print "done."

        bar = pbar.tqdm(total=int(9e9))
        subjects = 0
        rejected_dirs = []

        with open(fold_path+os.sep+train_file, "a+") as train, \
                open(fold_path + os.sep + test_file, "a+") as test:

            for root, dirs, files in os.walk(self.dataset_path):

                if root != self.dataset_path:
                    bar.set_description("Dir-->" + os.path.basename(root))
                    # seed = fold*i
                    random.Random().shuffle(files)

                    bar.update()

                    if len(files) - 1 < num_train:
                        rejected_dirs.append(os.path.basename(root))
                        continue

                    subjects += 1

                    train_split = files[0:num_train]
                    test_split = files[num_train:]

                    train_sample = os.path.basename(root) + ", " + \
                        reduce(lambda l, s: l + ", " +
                               s, train_split) + "\n"

                    test_sample = os.path.basename(root) + ", " + \
                        reduce(lambda l, s: l + ", " +
                               s, test_split) + "\n"

                    train.write(train_sample)
                    test.write(test_sample)

                elif root == self.dataset_path:
                    bar.total = len(dirs)

            bar.close()

        if len(rejected_dirs) > 0:
            print "The following directories were rejected: ", rejected_dirs
        print "We have", subjects, "subjects in our dataset."

    def load_data(self, is_train, num_train, fold):
        """
        Gets the test or train data

        :param is_train: a flag to indicate whether we need training data or test data
        :type is_train: bool
        :param fold: the fold for which data needs to be loaded
        :type fold: int
        :param num_train: the subdirectory indicating number of training samples per subject
        :type num_train: int
        :return test data: a tuple of a vector of faces and it's corresponding labels
        :rtype test data: (numpy.ndarray, numpy.ndarray)
        """

        if not os.path.exists(self.dataset_split):
            raise Exception(
                "Dataset test-train split has not been generated, yet! Please run split() before loading data.")

        if fold is None:
            raise Exception("Fold identifier is uspecified!")

        if num_train is None:
            raise Exception("Training sample identifier unspecified!")

        if is_train is None:
            raise Exception("Train/Test flag is unspecified!")

        suffix = None
        if is_train:
            suffix = "train.csv"
        else:
            suffix = "test.csv"

        csv_file = self.dataset_split + os.path.sep + \
            str(num_train) + os.path.sep + str(fold) + os.path.sep + suffix
        X = []
        y = []

        with open(csv_file, "r") as file:
            for line in file:
                parts = line.split("\n")
                parts = parts[0].split(",")
                label = parts[0]
                imgs = parts[1:]

                # print label, "imgs:", train_imgs

                images_path = self.dataset_path + os.path.sep + label
                for image in imgs:
                    img = cv2.imread(os.path.join(
                        images_path, image.strip()), cv2.IMREAD_GRAYSCALE)
                    # print img.shape
                    X.append(img)
                    y.append(int(label))

        y = np.array(y)
        # X = np.array(X)
        # print y.shape, X.shape

        return X, y


if __name__ == "__main__":
    dataset = Dataset("../../../datasets/norm_standard_att")
    folds = 3
    training_samples = [2, 5, 8]

    # dataset.clear_splits()
    # for training_sample in training_samples:
    #     print "Generating for", training_sample, "training samples per subject"
    #     for i in range(1, folds+1):
    #         print "Generating: Fold", i
    #         dataset.split(num_train=training_sample, fold=i)

    X_train, y_train = dataset.load_data(is_train=True, fold=1, num_train=2)
    X_test, y_test = dataset.load_data(is_train=False, fold=1, num_train=2)

    print len(X_train), y_train.shape, X_train[0].shape
