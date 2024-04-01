"""
File: main.py
Author: Chuncheng Zhang
Date: 2024-03-27
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-03-27 ------------------------
# Requirements and constants
import numpy as np
import mne
import random
import matplotlib.pyplot as plt

from module import logger
from module.constants import FIFF
from module.files import search_all_files
from module.load_data import JointData
from module.locate_sensors import JointData_SensorsPosition
from module.analysis_epochs import JointData_SensorsPosition_Epochs


# %% ---- 2024-03-27 ------------------------
# Function and class
def mk_inputs(subject_id: str = None):

    # --------------------
    # Search files
    files_table = search_all_files()
    print(f'Found files: {files_table}')

    # --------------------
    # Specific the subject_id and session_id
    # subject_id = 'S06_rawdata'
    subject_id = 'S09_20240110'

    # --------------------
    # Use the given subject_id & session_id
    # Or, random choice subject_id & session_id, if not given
    if subject_id is None:
        subject_id = random.choice(files_table['subject_id'])
        logger.debug(f'Random selected subject_id: {subject_id}')

    logger.debug(f'Selected subject_id: {subject_id}')
    group = files_table.groupby(['subject_id', 'experiment'])
    complex_experiment_files = group.get_group((subject_id, 'complex'))

    inputs = dict(
        subject_id=subject_id,
        noise_raw_path=group.get_group((subject_id, 'noise')).iloc[0]['path'],
        MI_raw_path=complex_experiment_files['path'].to_list()
    )

    logger.debug(f'Generated inputs: {inputs}')

    return inputs


# %% ---- 2024-03-27 ------------------------
# Play ground
if __name__ == '__main__':
    logger.info('**** Started ****')

    inputs = mk_inputs()
    logger.debug(f'Using inputs: {inputs}')

    # jd = JointData(inputs)
    jd = JointData_SensorsPosition_Epochs(inputs)

    jd.plot_positions()

    jd.plot_evoked()

    jd.compute_meg_eeg_signal_distance()

    jd.close_pdf()

    # fig = jd.plot_positions_3d()
    # fig.show()

    print('Done.')


# %% ---- 2024-03-27 ------------------------
# Pending


# %%
