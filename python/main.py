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

    # jd.plot_evoked()

    # jd.compute_meg_eeg_signal_distance()

    jd.close_pdf()

    # fig = jd.plot_positions_3d()
    # fig.show()

    print('Done.')

# %%

# %%

# %%

# %%
'''
jd.epochs.info['chs']

# %%
jd.raw.info
jd.epochs.info

# %%
fig = mne.viz.plot_sensors(jd.epochs.info, kind='3d', ch_type='mag')

# %%
fig = mne.viz.plot_sensors(
    jd.epochs.info, kind='topomap', ch_groups='position')
fig = mne.viz.plot_sensors(
    jd.epochs.info, kind='topomap', ch_groups='position', ch_type='mag')
fig = mne.viz.plot_sensors(
    jd.epochs.info, kind='topomap', ch_groups='position', ch_type='eeg')
# %%
fig = mne.viz.plot_sensors(jd.epochs.info, kind='3d',
                           ch_type='eeg', show_names=True)
fig = mne.viz.plot_sensors(
    jd.epochs.info, kind='topomap', ch_type='eeg', show_names=True)

# %%

# %%
# evoked.plot_joint()
jd.raw.ch_names

# %%
jd.raw.get_montage().dig

# %%
jd.mm.montage.dig

# %%
evoked = jd.epochs['1'].average()
evoked

# %%
fig = evoked.copy().pick('mag').plot_joint()
fig = evoked.copy().pick('eeg').plot_joint()

# %%
evoked.apply_proj()
evoked.save('evoked_avg.fif')

evoked = mne.Evoked('evoked_avg.fif')
evoked
# %%
evoked.ch_names

# %%
evoked.plot_joint()

# %%
[(key, value) for key, value in evoked.info.items()]

# %%
[(i, e['kind']) for i, e in enumerate(evoked.get_montage().dig)]
# %%
for ch in evoked.info['chs']:
    ch['kind'] = FIFF.FIFFV_POINT_EEG
    print(ch)


# %%
# jd.epochs.load_data()
jd.epochs.pick('mag')
# mne.viz.plot_sensors(jd.raw.info, ch_type='mag')
# mne.viz.plot_sensors(jd.raw.info, ch_type='eeg')

# %%
jd.epochs.pick_types()

# %%

# %%
montage = jd.raw.get_montage()
eegs = [e for e in jd.raw.ch_names if e.startswith('EEG')]
digs = [e for e in montage.dig if e['kind'] == FIFF.FIFFV_POINT_EEG]
for i, (name, dig) in enumerate(zip(eegs, digs)):
    print(i, name, dig)
    dig['r'] = i + np.array([1.0, 2.0, 3.0])

for e in montage.dig:
    print(e)

# %%
montage64 = mne.channels.make_standard_montage('biosemi64')
montage64
for e in montage64.dig:
    print(e)

# %%
for e in montage.dig:
    print(e)

# %%


def match_montages(src_montage, dst_montage):
    # src x trans = dst
    src = np.eye(4)
    dst = np.eye(4)

    src[:3, :3] = np.array(
        [e['r'] for e in src_montage.dig if e['kind'] == FIFF.FIFFV_POINT_CARDINAL])
    dst[:3, :3] = np.array(
        [e['r'] for e in dst_montage.dig if e['kind'] == FIFF.FIFFV_POINT_CARDINAL])

    return np.matmul(np.linalg.inv(src), dst)


def transform(mat, src):
    s = np.zeros((1, 4))
    s[0, :3] = src
    return np.matmul(s, mat)[0, :3].ravel()


montage64 = mne.channels.make_standard_montage('biosemi64')
montage = jd.raw.get_montage()
trans_mat = match_montages(montage64, montage)

for e in montage64.dig:
    print(e)
    e['r'] = transform(trans_mat, e['r']).ravel()

for e in montage64.dig:
    print(e)

montage64.ch_names

# %%
# FIFF = mne._fiff.constants.FIFF
FIFF.FIFFV_POINT_EEG

# %%
# jd.epochs_eeg, jd.epochs_meg

montage = jd.epochs_eeg.get_montage()
type(montage)
montage.get_positions()
montage.dig

# %%
montage = jd.raw.get_montage()
print(montage)
print(montage.get_positions())

# %%
montage = jd.raw.get_montage()
for e in montage.dig:
    print(e)
    print(e['r'])
    print(type(e['r']))
    e['r'] = np.array([1.0, 1.0, 1.0])

for e in montage.dig:
    print(e)
    # print(e['r'])

# %%
mne.channels.montage.DigMontage
# %%

# %% ---- 2024-03-27 ------------------------
# Pending
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# for event in jd.event_id:
#     _experiment = f'{event}-{jd.experiment_events[event]}'
#     print(_experiment)
#     meg = jd.get_meg_evoked(event).data
#     eeg = jd.get_eeg_evoked(event).data
#     print(meg.shape, eeg.shape)
#     coef = np.corrcoef(meg, eeg)
#     print(coef.shape)

#     dm = jd.distance_matrix.copy()
#     df = jd.distance_matrix.copy()

#     fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#     sns.heatmap(coef, ax=ax, cmap='RdBu').set_title(f'Corr-{_experiment}')

#     fig, axs = plt.subplots(2, 1, figsize=(8, 8))
#     arr = coef[-len(eeg):, :len(meg)]
#     df.loc[:] = arr
#     sns.heatmap(dm, ax=axs[0], cmap='cividis').set_title(f'Sensor-dist-{_experiment}')
#     sns.heatmap(df, ax=axs[1], cmap='RdBu').set_title(f'Signal-corr-{_experiment}')
#     plt.tight_layout()
#     plt.show()

# print('Done.')

# %%

# %%
# %% ---- 2024-03-27 ------------------------
# Pending


# %%
'''
