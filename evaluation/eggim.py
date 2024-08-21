import pandas as pd


def compute_eggim_patient(patient_df, target_variable):
    df = patient_df[['landmark', target_variable]].value_counts()
    # we use index[0] to assume first landmark/score pair is correct
    eggim_antrum_incisura = df['proximal ant lc'].index[0] + df['proximal ant gc/pw'].index[0] + df['incisura r'].index[0]
    eggim_body_1 = (df['distal body'].index[0] + df['upper body ant'].index[0]) / 2
    eggim_body_2 = (df['distal lc'].index[0]+ df['upper body r'].index[0]) / 2
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
