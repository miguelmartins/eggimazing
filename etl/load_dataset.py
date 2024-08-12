import json
import math
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit


class DatasetProcessor:
    def __init__(self, target_directory, image_extensions=('jpg'), annotation_extensions=('json'), id_prefix_size=10):
        self.target_directory = target_directory
        self.image_extensions = image_extensions
        self.annotation_extension = annotation_extensions
        # We assume that the patient id is encoded in the first id_prefix_size numbers of each file
        self.id_prefix_size = id_prefix_size
        self.dataset_dictionary = self._load_file_names()

    def _load_file_names(self):
        dataset_files = sorted(os.listdir(self.target_directory))
        json_names = [x for x in dataset_files if x.endswith('.json')]
        image_names = [x for x in dataset_files if x not in json_names]
        patient_ids = np.array([x[:self.id_prefix_size] for x in json_names])
        return {pid_: [[image_ for image_ in image_names if image_.startswith(pid_)],
                       [json_ for json_ in json_names if json_.startswith(pid_)]] for pid_ in patient_ids}

    # bbox, eggim in bbox, landmark
    def process_json(self, directory):
        with open(directory, 'r') as file:
            data = json.load(file)
        dict_parameters = {}
        for instance in data['instances']:
            if instance['className'] == 'EGGIM in the FULL Anatomical Location':
                dict_parameters['eggim_global'] = int(instance['attributes'][0]['name'])
            if instance['className'] == 'EGGIM in Target Area - Square':
                dict_parameters['eggim_square'] = int(instance['attributes'][0]['name'])
            if instance['className'] == 'Anatomical Location':
                dict_parameters['landmark'] = str(instance['attributes'][0]['name'])
            if instance["type"] == "bbox" and "points" in instance:
                points = instance["points"]
                left = points["x1"]
                top = points["y1"]
                right = points["x2"]
                bottom = points["y2"]
                # print("x1", left, "y1", top, "x2", right, "y2", bottom)
                dict_parameters['bbox'] = np.array([math.floor(left), math.floor(top), math.floor(right), math.floor(
                    bottom)])  # plt.imshow(np.array(image)[round(y1):round(y2), round(x1):round(x2), :])
        return dict_parameters

    def process(self, merge_eggim_square=False):
        dataset_info = []
        for patient, (images, jsons) in self.dataset_dictionary.items():
            for x, y in zip(images, jsons):
                annotation_data = self.process_json(os.path.join(self.target_directory, y))
                annotation_data['image_directory'] = os.path.join(self.target_directory, x)
                dataset_info.append(annotation_data)
        df = pd.DataFrame(dataset_info)
        if merge_eggim_square:
            df['eggim_square'] = df['eggim_square'].apply(lambda score: 0 if score == 0 else 1)
        return df

    def stratified_k_splits(X, y, k=5, train_size=0.7, val_size=0.15, test_size=0.15, random_state=None):
        assert train_size + val_size + test_size == 1.0, "The sum of train, val, and test sizes must be 1.0"
        # Create k StratifiedShuffleSplit instances
        sss = StratifiedShuffleSplit(n_splits=k, train_size=train_size, test_size=val_size + test_size,
                                     random_state=random_state)
        for train_idx, temp_idx in sss.split(X, y):
            X_train, X_temp = X[train_idx], X[temp_idx]
            y_train, y_temp = y[train_idx], y[temp_idx]
            # Split the temp set into validation and test sets
            sss_temp = StratifiedShuffleSplit(n_splits=1, train_size=val_size / (val_size + test_size),
                                              test_size=test_size / (val_size + test_size), random_state=random_state)
            val_idx, test_idx = next(sss_temp.split(X_temp, y_temp))
            yield train_idx, val_idx, test_idx


def crop_image(image, bbox, crop_height=224, crop_width=224):
    # Crop the image to the bounding box
    cropped_image = tf.image.crop_to_bounding_box(image, bbox[1], bbox[0], crop_height, crop_width)

    # Resize the cropped image to the desired size
    resized_image = tf.image.resize(cropped_image, [crop_height, crop_width])
    return resized_image


def load_and_preprocess_image(image_path, bbox):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = crop_image(image, bbox)
    return image


def get_data(image_dir, eggim_square_score, bbox, num_classes):
    bbox = tf.cast(bbox, dtype=tf.int32)
    x = tf.cast(load_and_preprocess_image(image_dir, bbox), dtype=tf.float32)
    if num_classes == 2:
        y = tf.cast(eggim_square_score, dtype=tf.float32)
    else:
        y = tf.cast(tf.one_hot(eggim_square_score, num_classes), dtype=tf.float32)
    return x, y


def get_tf_eggim_patch_dataset(df: pd.DataFrame, num_classes: int = 2):
    bboxes = np.stack(np.array(df['bbox'].values), axis=-1).T
    images = df['image_directory'].values
    eggim_square = df['eggim_square'].values

    # Assuming images, eggim_square, and bboxes are defined properly somewhere in your code.
    image_ds = tf.data.Dataset.from_tensor_slices(images)
    eggim_square_ds = tf.data.Dataset.from_tensor_slices(eggim_square)
    bboxes_ds = tf.data.Dataset.from_tensor_slices(bboxes)

    # Combine the datasets into a single dataset
    dataset = tf.data.Dataset.zip((image_ds, eggim_square_ds, bboxes_ds))

    dataset_processed = dataset.map(lambda img, score, bbox: get_data(img, score, bbox, num_classes),
                                    num_parallel_calls=tf.data.AUTOTUNE)
    return dataset_processed
