import os
import random
import shutil
import tqdm as pbar


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

        try:
            assert os.path.exists(os.path.realpath(dataset_path))
        except:
            print "Invalid dataset path!"

        dataset_split_path_suffix = "__train_test_split"

        self.dataset_path = os.path.realpath(dataset_path)
        self.dataset_split = os.path.basename(
            dataset_path)+dataset_split_path_suffix

        if os.path.exists(self.dataset_split):
            shutil.rmtree(self.dataset_split)
        os.mkdir(self.dataset_split)

    def split(self, num_train=2, fold=1):
        """
        Generates a train-test split based on the number of training samples

        :param num_train: number of training samples per split
        :type num_train: int
        :param fold: the fold the split belongs to
        :type fold: int
        """

        train_file = "train.csv"
        test_file = "test.csv"

        fold_path = self.dataset_split + os.sep + str(fold)
        if os.path.exists(fold_path):
            shutil.rmtree(fold_path)

        os.mkdir(fold_path)

        bar = pbar.tqdm(total=0)

        with open(fold_path+os.sep+train_file, "a+") as train, \
                open(fold_path + os.sep + test_file, "a+") as test:

            i = 0
            for root, dirs, files in os.walk(self.dataset_path):

                if root != self.dataset_path:
                    bar.set_description("Dir: "+os.path.basename(root)+"--> ")
                    seed = fold*i
                    random.Random().shuffle(files)

                    train_split = files[0:num_train]
                    test_split = files[num_train:]

                    train_sample = os.path.basename(root) + ", " + \
                        reduce(lambda l, s: l + ", " + s, train_split) + "\n"
                    test_sample = os.path.basename(root) + ", " + \
                        reduce(lambda l, s: l + ", " + s, test_split) + "\n"

                    train.write(train_sample)
                    test.write(test_sample)
                    i += 1
                    bar.update(i)
                else:
                    bar.total = len(dirs)

            bar.close()


if __name__ == "__main__":
    dataset = Dataset("../../../datasets/standard_att")
    folds = 5

    for i in range(1, folds+1):
        print "Generating 'Fold", i, "'\n"
        dataset.split(fold=i)
