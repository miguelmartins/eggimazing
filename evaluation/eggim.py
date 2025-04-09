import pandas as pd


def compute_eggim_patient_deprecated(patient_df, target_variable):
    df = patient_df[['landmark', target_variable]].value_counts()
    # we use index[0] to assume first landmark/score pair is correct
    eggim_antrum_incisura = df['proximal ant lc'].index[0] + df['proximal ant gc/pw'].index[0] + df['incisura r'].index[
        0]
    eggim_body_1 = (df['distal body'].index[0] + df['upper body ant'].index[0]) / 2
    eggim_body_2 = (df['distal lc'].index[0] + df['upper body r'].index[0]) / 2
    return eggim_antrum_incisura + eggim_body_1 + eggim_body_2


def replace_landmark_name(landmark):
    new_landmark_name = {'ii': 'distal body',
                         'ix': 'distal lc',
                         'vi': 'proximal ant lc',
                         'vii': 'proximal ant gc/pw',
                         'viii': 'incisura r',
                         'x': 'upper body r',
                         'xii': 'upper body ant'}
    landmark_number = landmark.split('.')[0]
    return new_landmark_name[landmark_number]


def compute_eggim_patient(patient_df, target_variable):
    '''
    Note that one should run replace_landmark_name first
    :param patient_df:
    :param target_variable:
    :return:
    '''
    df = patient_df[['landmark', target_variable]].value_counts()

    # Assume first landmark/score pair is correct
    eggim_antrum_incisura = (
            df['proximal ant lc'].index[0] +
            df['proximal ant gc/pw'].index[0] +
            df['incisura r'].index[0]
    )

    # Compute eggim_body_1
    body_1_values = []
    if 'distal body' in df:
        body_1_values.append(df['distal body'].index[0])
    if 'upper body ant' in df:
        body_1_values.append(df['upper body ant'].index[0])

    if body_1_values:
        eggim_body_1 = sum(body_1_values) / len(body_1_values)

    # Compute eggim_body_2
    body_2_values = []
    if 'distal lc' in df:
        body_2_values.append(df['distal lc'].index[0])
    if 'upper body r' in df:
        body_2_values.append(df['upper body r'].index[0])

    if body_2_values:
        eggim_body_2 = sum(body_2_values) / len(body_2_values)

    return eggim_antrum_incisura + eggim_body_1 + eggim_body_2


def get_eggim_df(df, y_preds, patient_ids):
    '''
    Take the set of ordinal predictions and compute eggim_score for each patient.
    A dataframe with ground truth square and global eggim, as well as the predicted eggim is returned.
    :param df: the target dataframe
    :param y_preds: ordinal predictions
    :param patient_ids: set of test patient ids
    :return: A dataframe with eggim_square, eggim_global, and eggim_pred as columns
    '''
    eggim_scores = {}
    for i, patient_id in enumerate(patient_ids):
        df_patient = df[df['patient_id'] == patient_id]
        df_patient['pred'] = y_preds[i]
        eggim_square = compute_eggim_patient(df_patient,
                                             target_variable='eggim_square')
        eggim_global = compute_eggim_patient(df_patient,
                                             target_variable='eggim_global')
        eggim_pred = compute_eggim_patient(df_patient,
                                           target_variable='pred')
        eggim_scores[patient_ids[i]] = {'eggim_square': eggim_square, 'eggim_global': eggim_global,
                                        'eggim_pred': eggim_pred}
    return pd.DataFrame(eggim_scores).T
