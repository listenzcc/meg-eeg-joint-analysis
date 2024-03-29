"""
File: analysis_epochs.py
Author: Chuncheng Zhang
Date: 2024-03-28
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Analysis on the epochs of the meg- and eeg-data

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-03-28 ------------------------
# Requirements and constants
import mne
import contextlib

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm.auto import tqdm

from . import logger
from .locate_sensors import JointData_SensorsPosition


# %% ---- 2024-03-28 ------------------------
# Function and class
class JointData_SensorsPosition_Epochs(JointData_SensorsPosition):
    evoked_meg = {}
    evoked_eeg = {}

    def __init__(self, obj: dict):
        super().__init__(obj)

    def find_nearest_meg_sensors(self, eeg_sensor_name: str, n: int = 5):
        dm = self.distance_matrix.copy()
        se = dm.loc[eeg_sensor_name]
        se = se.sort_values(ascending=True)
        return se.index[:n].tolist()

    def get_meg_evoked(self, event: str):
        # --------------------
        # Requested evoked exists, just use it
        evoked = self.evoked_meg.get(event)
        if evoked is not None:
            return evoked

        # --------------------
        # Requested evoked does not exist, make it
        # Try to read it from cached file,
        # Or compute it
        p = self.cache.get_path(f'meg-evoked-{event}-ave.fif')
        try:
            evoked = mne.Evoked(p)
            logger.debug(f'Loaded meg evoked from {p}')
        except Exception as err:
            logger.warning(f'Invalid cached file: {p}, {err}')
            evoked = self.epochs_meg[event].average()
            evoked.apply_proj()
            evoked.save(p, overwrite=True)
            logger.debug(f'Saved evoked: {p}')
            logger.debug(f'Averaged meg evoked: {evoked}')
        self.evoked_meg[event] = evoked.pick(picks='mag', exclude='bads')
        return evoked

    def get_eeg_evoked(self, event: str):
        # --------------------
        # Requested evoked exists, just use it
        evoked = self.evoked_eeg.get(event)
        if evoked is not None:
            return evoked

        # --------------------
        # Requested evoked does not exist, make it
        # Try to read it from cached file,
        # Or compute it
        p = self.cache.get_path(f'eeg-evoked-{event}-ave.fif')
        try:
            evoked = mne.Evoked(p)
            logger.debug(f'Loaded eeg evoked from {p}')
        except Exception as err:
            logger.warning(f'Invalid cached file: {p}, {err}')
            evoked = self.epochs_eeg[event].average()
            evoked.apply_proj()
            evoked.save(p, overwrite=True)
            logger.debug(f'Saved evoked: {p}')
            logger.debug(f'Averaged meg evoked: {evoked}')
        self.evoked_eeg[event] = evoked.pick(picks='eeg', exclude='bads')
        return evoked


    def compute_meg_eeg_signal_distance(self):
        figs = []
        for event in self.event_id:
            _experiment = f'{event}-{self.experiment_events[event]}'
            print(_experiment)
            meg = self.get_meg_evoked(event).data
            eeg = self.get_eeg_evoked(event).data
            print(meg.shape, eeg.shape)
            coef = np.corrcoef(meg, eeg)
            print(coef.shape)

            dm = self.distance_matrix.copy()
            df = self.distance_matrix.copy()

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            sns.heatmap(coef, ax=ax, cmap='RdBu').set_title(f'Corr-{_experiment}')
            figs.append(fig)

            fig, axs = plt.subplots(2, 1, figsize=(8, 8))
            arr = coef[-len(eeg):, :len(meg)]
            df.loc[:] = arr
            sns.heatmap(dm, ax=axs[0], cmap='cividis').set_title(f'Sensor-dist-{_experiment}')
            sns.heatmap(df, ax=axs[1], cmap='RdBu').set_title(f'Signal-corr-{_experiment}')
            figs.append(fig)

        for fig in figs:
            with contextlib.suppress(Exception):
                fig.tight_layout()
            self.append_pdf_fig(fig)

        return figs

    def plot_evoked(self):
        epochs_eeg = self.epochs_eeg
        epochs_meg = self.epochs_meg

        # --------------------
        # Set picks for meg&eeg data
        eeg_picks = ['C3', 'Cz', 'C4']
        meg_picks = []
        for e in eeg_picks:
            meg_picks += self.find_nearest_meg_sensors(e)
        meg_picks = list(set(meg_picks))
        logger.debug(f'Picks eeg&meg sensors: {eeg_picks}, {meg_picks}')

        figs = []

        for event in tqdm(self.event_id, 'Plotting evoked...'):
            _experiment = f'{event}-{self.experiment_events[event]}'
            logger.debug(f'Plotting experiment: {_experiment}')

            figs.extend(tqdm([
                mne.viz.plot_evoked_joint(
                    self.get_meg_evoked(event), exclude='bads', show=False, title=f'MEG-{_experiment}'),

                mne.viz.plot_evoked_joint(
                    self.get_eeg_evoked(event), exclude='bads', show=False, title=f'EEG-{_experiment}'),

                epochs_meg[event].plot_image(
                    picks=meg_picks, combine='mean', show=False, title=f'MEG-{_experiment}'),

                epochs_eeg[event].plot_image(
                    picks=eeg_picks, combine='mean', show=False, title=f'EEG-{_experiment}'),
            ]))

        figs.extend(tqdm([
            epochs_meg.plot_image(
                picks=meg_picks, combine='mean', show=False, title='MEG'),

            epochs_eeg.plot_image(
                picks=eeg_picks, combine='mean', show=False, title='EEG')
        ]))

        for fig in figs:
            with contextlib.suppress(Exception):
                fig.tight_layout()
            self.append_pdf_fig(fig)

        return figs


# %% ---- 2024-03-28 ------------------------
# Play ground


# %% ---- 2024-03-28 ------------------------
# Pending


# %% ---- 2024-03-28 ------------------------
# Pending
