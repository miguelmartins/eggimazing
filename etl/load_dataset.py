import json
import math
import numpy as np
import os
import pandas as pd
import re
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit, LeavePGroupsOut
from sklearn.utils import resample


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
            if instance['className'] == 'Comments':
                if not instance['attributes']:  # check if list is empty
                    continue
                else:
                    id_ = str(instance['attributes'][0]['name'])
                    id_ = re.split(r'[ \n]+', id_)[0]
                    if id_.startswith('PT'):  # This is necessary to mark the patients from togas
                        dict_parameters['patient_id'] = id_
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

    def process(self, merge_eggim_square=False, merge_eggim_global=False):
        dataset_info = []
        for patient_id, (images, jsons) in self.dataset_dictionary.items():
            for x, y in zip(images, jsons):
                annotation_data = self.process_json(os.path.join(self.target_directory, y))
                annotation_data['image_directory'] = os.path.join(self.target_directory, x)
                if 'patient_id' not in annotation_data:
                    annotation_data['patient_id'] = patient_id
                dataset_info.append(annotation_data)
        df = pd.DataFrame(dataset_info)
        if merge_eggim_square:
            df['eggim_square'] = df['eggim_square'].apply(lambda score: 0 if score == 0 else 1)
        if merge_eggim_global:
            df['eggim_global'] = df['eggim_global'].apply(lambda score: 0 if score == 0 else 1)
        return df

    @staticmethod
    def augment_dataframe_stratified(df1, df2, target_column='eggim_square'):
        # Step 1: Calculate the distribution of eggim_square in df1
        distribution_df1 = df1[target_column].value_counts(normalize=True)

        # Step 2: Sample from df2 according to the distribution in df1
        sampled_df2 = pd.DataFrame()
        for category, proportion in distribution_df1.items():
            # Number of samples for this category
            n_samples = int(len(df2) * proportion)
            # Filter df2 for the specific category
            df2_category = df2[df2[target_column] == category]

            # Sample from this category
            sampled_category = resample(df2_category, n_samples=n_samples, replace=True, random_state=42)

            # Append to the sampled_df2
            sampled_df2 = pd.concat([sampled_df2, sampled_category])

        # Step 3: Combine sampled_df2 with df1
        return pd.concat([df1, sampled_df2])

    @staticmethod
    def single_ds_patient_k_group_split(df_target,
                                        k=5,
                                        train_size=0.7,
                                        test_size=0.3,
                                        internal_train_size=0.5,
                                        target_column='eggim_square',
                                        random_state=None):
        assert train_size + test_size == 1.0
        assert (0 < internal_train_size) and (internal_train_size < 1)

        # Create k StratifiedShuffleSplit instances
        gss = GroupShuffleSplit(n_splits=k, train_size=train_size, test_size=test_size,
                                random_state=random_state)
        groups = df_target['patient_id'].values
        X = df_target.drop(columns=[target_column])
        y = df_target[target_column]
        for temp_idx, test_idx in gss.split(X, y, groups):
            df_temp = df_target.iloc[temp_idx]
            df_test = df_target.iloc[test_idx]

            X_train = df_temp.drop(columns=[target_column])
            y_train = df_temp[target_column]
            # Split the temp set into validation and test sets
            sss_temp = StratifiedShuffleSplit(n_splits=1, train_size=internal_train_size,
                                              test_size=1. - internal_train_size, random_state=random_state)
            train_idx, val_idx = next(sss_temp.split(X_train, y_train))

            df_train = df_temp.iloc[train_idx]
            df_val = df_temp.iloc[val_idx]

            yield df_train, df_val, df_test

    @staticmethod
    def naive_patient_k_group_split(df_target,
                                    df_extra,
                                    k=5,
                                    train_size=0.7,
                                    test_size=0.3,
                                    internal_train_size=0.5,
                                    target_column='eggim_square',
                                    random_state=None):
        assert train_size + test_size == 1.0
        assert (0 < internal_train_size) and (internal_train_size < 1)
        '''
        Train and validation are from a stratified split from TOGAS+IPO. This means
        that we have no control on how many patients from TOGAS are in each validation set
        '''

        # Create k StratifiedShuffleSplit instances
        gss = GroupShuffleSplit(n_splits=k, train_size=train_size, test_size=test_size,
                                random_state=random_state)
        groups = df_target['patient_id'].values
        X = df_target.drop(columns=[target_column])
        y = df_target[target_column]
        for temp_idx, test_idx in gss.split(X, y, groups):
            df_temp = df_target.iloc[temp_idx]
            df_test = df_target.iloc[test_idx]
            df_joint = pd.concat([df_temp, df_extra], axis=0)

            X_train = df_joint.drop(columns=[target_column])
            y_train = df_joint[target_column]
            # Split the temp set into validation and test sets
            sss_temp = StratifiedShuffleSplit(n_splits=1, train_size=internal_train_size,
                                              test_size=1. - internal_train_size, random_state=random_state)
            train_idx, val_idx = next(sss_temp.split(X_train, y_train))

            df_train = df_joint.iloc[train_idx]
            df_val = df_joint.iloc[val_idx]

            yield df_train, df_val, df_test

    @staticmethod
    def patient_k_group_split(df_target,
                              df_extra,
                              k=5,
                              train_size=0.7,
                              test_size=0.3,
                              internal_train_size=0.5,
                              target_variable='eggim_square',
                              random_state=None):
        """
        Togas is subject to 2 splits. The first is a group split that generates k-folds of non-identical
        sets [temp_togas, test_togas]_k. Then each  temp_togas_i (i from 1 to k) will be stratified into inter_train_size% train
        and validation [train_togas, val_togas, test_togas]_k.
        We will then SUBSAMPLE IPO with respect to the target variable of train_togas to augment train_togas_i.
        This ensures that validation comes only from togas
        and that the representative of distribution of the splits are representative with regards to the target variable
         distribtion in df_target.
        """
        assert train_size + test_size == 1.0
        assert (0 < internal_train_size) and (internal_train_size < 1)
        # Create k StratifiedShuffleSplit instances
        gss = GroupShuffleSplit(n_splits=k, train_size=train_size, test_size=test_size,
                                random_state=random_state)
        groups = df_target['patient_id'].values
        X = df_target.drop(columns=[target_variable])
        y = df_target[target_variable]
        for temp_idx, test_idx in gss.split(X, y, groups):
            df_temp = df_target.iloc[temp_idx]
            df_test = df_target.iloc[test_idx]

            X_temp = df_temp.drop(columns=[target_variable])
            y_temp = df_temp[target_variable]
            sss_temp = StratifiedShuffleSplit(n_splits=1, train_size=internal_train_size,
                                              test_size=1. - internal_train_size, random_state=random_state)
            train_idx, val_idx = next(sss_temp.split(X_temp, y_temp))

            df_train = df_target.iloc[train_idx]
            df_val = df_target.iloc[val_idx]
            df_train = DatasetProcessor.augment_dataframe_stratified(df_train,
                                                                     df_extra,
                                                                     target_column=target_variable)
            df_train = pd.concat([df_train, df_extra], axis=0)

            yield df_train, df_val, df_test

    @staticmethod
    def single_ds_patient_wise_split(df_target,
                                     patients_ids,
                                     internal_train_size=0.5,
                                     target_variable='eggim_square',
                                     random_state=None):

        assert (0 < internal_train_size) and (internal_train_size < 1)
        for patient_id in patients_ids:
            test_frames_rows = df_target['patient_id'] == patient_id

            df_test = df_target.loc[test_frames_rows]
            df_temp = df_target.loc[~test_frames_rows]
            X_temp = df_temp.drop(columns=[target_variable])
            y_temp = df_temp[target_variable]
            sss_temp = StratifiedShuffleSplit(n_splits=1, train_size=internal_train_size,
                                              test_size=1. - internal_train_size, random_state=random_state)
            train_idx, val_idx = next(sss_temp.split(X_temp, y_temp))

            df_train = df_target.iloc[train_idx]
            df_val = df_target.iloc[val_idx]

            yield df_train, df_val, df_test

    @staticmethod
    def patient_wise_split(df_target,
                           df_extra,
                           patients_ids,
                           internal_train_size=0.5,
                           target_variable='eggim_square',
                           random_state=None):

        assert (0 < internal_train_size) and (internal_train_size < 1)
        for patient_id in patients_ids:
            test_frames_rows = df_target['patient_id'] == patient_id

            df_test = df_target.loc[test_frames_rows]
            df_temp = df_target.loc[~test_frames_rows]
            X_temp = df_temp.drop(columns=[target_variable])
            y_temp = df_temp[target_variable]
            sss_temp = StratifiedShuffleSplit(n_splits=1, train_size=internal_train_size,
                                              test_size=1. - internal_train_size, random_state=random_state)
            train_idx, val_idx = next(sss_temp.split(X_temp, y_temp))

            df_train = df_target.iloc[train_idx]
            df_val = df_target.iloc[val_idx]
            df_train = DatasetProcessor.augment_dataframe_stratified(df_train,
                                                                     df_extra,
                                                                     target_column=target_variable)
            df_train = pd.concat([df_train, df_extra], axis=0)

            yield df_train, df_val, df_test


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


def load_image(image_path, resize_height=224, resize_width=224):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (resize_height, resize_width))
    return image


def get_data_patch(image_dir, eggim_square_score, bbox, num_classes, augmentation_fn=None,
                   preprocess_fn=tf.image.per_image_standardization):
    bbox = tf.cast(bbox, dtype=tf.int32)
    x = tf.cast(load_and_preprocess_image(image_dir, bbox), dtype=tf.float32)
    x = preprocess_fn(x)
    if augmentation_fn is not None:
        x = augmentation_fn(x)
    if num_classes == 2:
        y = tf.cast(eggim_square_score, dtype=tf.float32)
    else:
        y = tf.one_hot(tf.cast(eggim_square_score, dtype=tf.int32), num_classes)
    return x, y


def get_data(image_dir, eggim_score, num_classes, augmentation_fn=None,
             preprocess_fn=tf.image.per_image_standardization):
    x = tf.cast(load_image(image_dir), dtype=tf.float32)
    x = preprocess_fn(x)
    if augmentation_fn is not None:
        x = augmentation_fn(x)
    if num_classes == 2:
        y = tf.cast(eggim_score, dtype=tf.float32)
    else:
        y = tf.one_hot(tf.cast(eggim_score, dtype=tf.int32), num_classes)
    return x, y


def get_tf_eggim_patch_dataset(df: pd.DataFrame, num_classes: int = 2, augmentation_fn=None,
                               preprocess_fn=tf.image.per_image_standardization):
    bboxes = np.stack(np.array(df['bbox'].values), axis=-1).T
    images = df['image_directory'].values
    eggim_square = df['eggim_square'].values

    # Assuming images, eggim_square, and bboxes are defined properly somewhere in your code.
    image_ds = tf.data.Dataset.from_tensor_slices(images)
    eggim_square_ds = tf.data.Dataset.from_tensor_slices(eggim_square)
    bboxes_ds = tf.data.Dataset.from_tensor_slices(bboxes)

    # Combine the datasets into a single dataset
    dataset = tf.data.Dataset.zip((image_ds, eggim_square_ds, bboxes_ds))

    dataset_processed = dataset.map(lambda img, score, bbox: get_data_patch(img,
                                                                            score,
                                                                            bbox,
                                                                            num_classes,
                                                                            augmentation_fn=augmentation_fn,
                                                                            preprocess_fn=preprocess_fn),
                                    num_parallel_calls=tf.data.AUTOTUNE)
    return dataset_processed


def get_tf_eggim_full_image_dataset(df: pd.DataFrame, num_classes: int = 2, augmentation_fn=None,
                                    preprocess_fn=tf.image.per_image_standardization):
    images = df['image_directory'].values
    eggim_score = df['eggim_global'].values

    # Assuming images, eggim_square, and bboxes are defined properly somewhere in your code.
    image_ds = tf.data.Dataset.from_tensor_slices(images)
    eggim_ds = tf.data.Dataset.from_tensor_slices(eggim_score)

    # Combine the datasets into a single dataset
    dataset = tf.data.Dataset.zip((image_ds, eggim_ds))

    dataset_processed = dataset.map(lambda img, score: get_data(img,
                                                                score,
                                                                num_classes,
                                                                augmentation_fn=augmentation_fn,
                                                                preprocess_fn=preprocess_fn),
                                    num_parallel_calls=tf.data.AUTOTUNE)
    return dataset_processed


def get_valid_patient_ids_deprecated(dataframe, p_ids):
    lands = [x.split('.')[0] for x in np.unique(dataframe.landmark).squeeze()]
    valid_pids = []
    for p_id in p_ids:
        if set([x.split('.')[0] for x in dataframe[dataframe.patient_id == p_id].landmark]) == set(lands):
            valid_pids.append(p_id)
    return valid_pids


def get_valid_patient_ids(dataframe):
    df = dataframe.copy()
    p_ids = list(set(df['patient_id']))
    valid_patients = []
    for p_id in p_ids:
        p_lands = np.unique(df[df.patient_id == p_ids[0]].landmark).squeeze()
        p_lands = [x.split('.')[0] for x in p_lands]
        if 'ii' in p_lands or 'xii' in p_lands:
            if 'ix' in p_lands or 'x' in p_lands:
                if 'vi' in p_lands and \
                        'vii' in p_lands and \
                        'viii' in p_lands:
                    valid_patients.append(p_id)
    return valid_patients
