import numpy as np

def compute_eggim_patient(patient_df, target_variable):
    eggim_landmarks = ['proximal ant lc',
     'proximal ant gc/pw',
     'incisura r',
     'distal body',
     'upper body ant',
     'distal lc',
     'upper body r']

    df = patient_df[['landmark', target_variable]].value_counts()
    # we use index[0] to assume first landmark/score pair is correct
    eggim_antrum_incisura = df['proximal ant lc'].index[0] + df['proximal ant gc/pw'].index[0] + df['incisura r'].index[0]
    eggim_body_1 = (df['distal body'].index[0] + df['upper body ant'].index[0]) / 2
    eggim_body_2 = (df['distal lc'].index[0]+ df['upper body r'].index[0]) / 2
    return eggim_antrum_incisura + eggim_body_1 + eggim_body_2



def get_eggim_df(df):
    eggim_scores = {}
    patient_ids = np.unique(df['patient_id'].values)
    for i in range(len(patient_ids)):
        try:
            eggim_square = compute_eggim_patient(df[df.patient_id == patient_ids[i]],
                                          target_variable='eggim_square')
            eggim_global = compute_eggim_patient(df[df.patient_id == patient_ids[i]],
                                          target_variable='eggim_global')
            eggim_scores[patient_ids[i]] = {'eggim_square': eggim_square, 'eggim_global': eggim_global}
        except:
            continue
    return eggim_scores