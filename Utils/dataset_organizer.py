import os
import argparse
import shutil

if __name__ == '__main__':
    dataset = None

    parser = argparse.ArgumentParser(
        description="Standardize the dataset structure as directories for subjects containing their faces.")
    parser.add_argument("dataset", help="Path to the dataset.")

    args = parser.parse_args()

    standard_dataset_path = 'standard_' + args.dataset.split(os.path.sep)[-1]
    if os.path.exists(standard_dataset_path):
        shutil.rmtree(standard_dataset_path)
    else:
        os.makedirs(standard_dataset_path)

    for root, dir, files in os.walk(args.dataset):
        for name in files:
            if name.endswith('.png'):
                parts = (name.split('.')[0]).split('_')
                subject = parts[0]
                face = parts[1]

                subject_path = standard_dataset_path + os.sep + subject
                face_path = subject_path + os.sep + name

                if not os.path.exists(subject_path):
                    os.makedirs(subject_path)

                shutil.copy(args.dataset + os.sep + name,
                            subject_path)

                os.rename(face_path, subject_path + os.sep + face + '.png')
