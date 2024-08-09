from etl.load_dataset import DatasetProcessor


def test_number_of_patients():
    target_dir = '../test_files/TOGETHER'
    dp = DatasetProcessor(target_dir)
    assert len(dp.dataset_dictionary.keys()) == 4


def test_unique_json_failes():
    target_dir = '../test_files/TOGETHER'
    dp = DatasetProcessor(target_dir)
    dp_dict = dp.dataset_dictionary
    number_of_jsons = [dp_dict[key_][1] for key_ in dp_dict.keys()]
    assert len(number_of_jsons) == len(dp.dataset_dictionary.keys()) == 4


def test_multiple_images_single_patient():
    target_dir = '../test_files/TOGETHER'
    dp = DatasetProcessor(target_dir)
    dp_dict = dp.dataset_dictionary
    patient_multiple_frame_id = '2024012310'
    assert len(dp_dict[patient_multiple_frame_id][0]) == 2


def main():
    test_number_of_patients()
    test_unique_json_failes()
    test_multiple_images_single_patient()


if __name__ == '__main__':
    main()
