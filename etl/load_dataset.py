import os
import numpy as np


class DatasetProcessor:
    def __init__(self, target_directory, image_extensions=('jpg'), annotation_extensions=('json'), id_prefix_size=10):
        self.target_directory = target_directory
        self.image_extensions = image_extensions
        self.annotation_extension = annotation_extensions
        # We assume that the patient id is encoded in the first id_prefix_size numbers of each file
        self.id_prefix_size = id_prefix_size
        self.dataset_dictionary = self._load_file_names()

    def _load_file_names(self):
        dataset_files = os.listdir(self.target_directory)
        json_names = [x for x in dataset_files if x.endswith('.json')]
        image_names = [x for x in dataset_files if x not in json_names]
        patient_ids = np.array([x[:self.id_prefix_size] for x in json_names])
        return {pid_: [[image_ for image_ in image_names if image_.startswith(pid_)],
                       [json_ for json_ in json_names if json_.startswith(pid_)]] for pid_ in patient_ids}
