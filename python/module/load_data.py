"""
File: raw.py
Author: Chuncheng Zhang
Date: 2024-03-27
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Load the raw object from the meg-eeg data

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-03-27 ------------------------
# Requirements and constants
import re
import mne
import contextlib
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

from . import logger


# %% ---- 2024-03-27 ------------------------
# Function and class

# Only the first 35 out of 64 channels are used
# 35 = 5 x 7
channels_in_order_5x7 = '''
f5,  f3,  f1,  fz,  f2,  f4,  f6,
fc5, fc3, fc1, fcz, fc2, fc4, fc6,
c5,  c3,  c1,  cz,  c2,  c4,  c6,
cp5, cp3, cp1, cpz, cp2, cp4, cp6,
p5,  p3,  p1,  pz,  p2,  p4,  p6
'''

montage64_name = 'biosemi64'


def _ch_name(s: str) -> str:
    # c4 -> C4
    # cz -> Cz
    s = s.strip()
    return s[:-1].upper() + s[-1]


def _reset_eeg_montage(epochs):
    # Load everything
    channels_in_order = [
        _ch_name(e)
        for e in channels_in_order_5x7.split(',')
    ]
    montage64 = mne.channels.make_standard_montage(montage64_name)

    for name in channels_in_order:
        assert name in montage64.ch_names, f'Unknown channel: {name}'
    assert len(
        epochs.ch_names) == 35, f'Incorrect number of channels, {epochs}'

    rename = {epochs.ch_names[j]: channels_in_order[j] for j in range(35)}
    epochs.rename_channels(rename)

    epochs.set_montage(montage64, on_missing='warn')

    return epochs

class MyCache(object):
    root = Path('./cache')
    path = None

    def __init__(self, uid:str):
        self.path = self.root.joinpath(uid)

    def get_path(self, subpath):
        path = self.path.joinpath(subpath)
        path.parent.mkdir(exist_ok=True, parents=True)
        return path


class JointData(object):
    inputs = None
    cache = None
    empty_room_noise_path = None
    pdf_pages = None
    pdf_pages_path = None

    raw = None
    events = None
    event_id = None
    epochs_meg = None
    epochs_eeg = None

    experiment_events = {
        '1': 'Hand',  # '手',
        '2': 'Wrist',  # '腕',
        '3': 'Elbow',  # '肘',
        '4': 'Shoulder',  # '肩',
        '5': 'Rest',  # '静息',
    }

    def __init__(self, obj: dict):
        self.inputs = obj
        self.init_cache()
        self.open_pdf()
        self.load_raw()
        self.add_empty_room_noise_proj()
        self.get_epochs()

    def init_cache(self):
        uid = self.inputs['subject_id']
        self.cache = MyCache(uid)

    def open_pdf(self):
        subject_id = self.inputs['subject_id']
        path = Path(f'./pdf-pages/{subject_id}/pages.pdf')
        path.parent.mkdir(exist_ok=True, parents=True)
        pdf_pages = PdfPages(path)
        self.pdf_pages = pdf_pages
        self.pdf_pages_path = path
        logger.debug(f'Created PDF pages in {path}')

        fig = plt.figure(figsize=(8, 4))
        fig.clf()
        fig.text(
            0.5, 0.2, subject_id,
            transform=fig.transFigure, size=24, ha='center')
        fig.text(
            0.5, 0.6, datetime.now(),
            transform=fig.transFigure, size=24, ha='center')
        self.append_pdf_fig(fig)

    def close_pdf(self):
        with contextlib.suppress(Exception):
            self.pdf_pages.close()
        logger.debug(f'Closed pdf_pages: {self.pdf_pages}')
        logger.debug(f'Saved pdf_pages: {self.pdf_pages_path}')

    def append_pdf_fig(self, fig: plt.Figure):
        def _append_fig(fig):
            try:
                self.pdf_pages.savefig(fig)
                logger.debug(f'Saved figure to pdf: {fig}')
            except Exception as err:
                logger.error(f'Failed to save figure: {fig}')

        if isinstance(fig, list):
            for f in fig:
                _append_fig(f)
        else:
            _append_fig(fig)

    def load_raw(self):
        # It should be a list of path
        path_array = self.inputs.get('MI_raw_path')
        if not isinstance(path_array, list):
            path_array = [path_array]

        raws = []
        dev_head_t = None
        for path in path_array:
            r = mne.io.read_raw_ctf(path)
            if dev_head_t is None:
                dev_head_t = r.info['dev_head_t']
            r.info['dev_head_t'] = dev_head_t

            # sfreq=100
            # r.resample(sfreq=sfreq, n_jobs=32)
            # logger.debug(f'Resample the raw to {sfreq}')

            raws.append(r)
            logger.debug(f'Loaded raw from {path}')

        raw = mne.concatenate_raws(raws)
        # mne.preprocessing.find_bad_channels_maxwell(raw)
        # mne.preprocessing.maxwell_filter(raw)
        events, event_id = mne.events_from_annotations(raw)

        self.path_array = path_array
        self.raw = raw
        self.events = events
        self.event_id = event_id

        logger.debug(f'Loaded raw: {path_array}, {raw}')


        return raw, events, event_id

    def add_empty_room_noise_proj(self):
        try:
            path = self.inputs['noise_raw_path']
            em_raw = mne.io.read_raw_ctf(path)
            em_projs = mne.compute_proj_raw(em_raw, n_mag=3, n_eeg=3)
            self.raw.add_proj(em_projs)
            self.raw.apply_proj(verbose=True)
            self.empty_room_noise_path = path
            logger.debug(f'Added empty room noise projs: {em_projs}')
        except Exception as err:
            logger.error(f'Invalid empty room noise file: {path}, {err}')

    def get_epochs(self):
        raw = self.raw
        events = self.events
        event_id = self.event_id


        # --------------------
        kwargs = dict(
            tmin=-1,  # Starts from -1.0 seconds
            tmax=4,  # Ends at 4.0 seconds
            decim=20,  # Down-samples from 1200 Hz to 60 Hz
            detrend=0,  # Remove DC
        )

        # --------------------
        # Reject template
        reject = dict(
            grad=4000e-13,  # unit: T / m (gradiometers)
            mag=4e-12,      # unit: T (magnetometers)
            eeg=40e-6,      # unit: V (EEG channels)
            eog=250e-6      # unit: V (EOG channels)
        )

        # --------------------
        # MEG
        unit_1ft = 1e-15  # Teslas
        epochs_meg = mne.Epochs(raw, events, event_id, reject={
                                'mag': 4000 * unit_1ft}, picks='mag', **kwargs)

        # --------------------
        # EEG
        unit_1uv = 1e-6  # Volts
        picks = [e for e in raw.ch_names if e.startswith('EEG')][:35]
        epochs_eeg = mne.Epochs(raw, events, event_id, reject={
                                'eeg': 400 * unit_1uv}, picks=picks, **kwargs)
        # The empty-room noise does not contain the EEG components
        epochs_eeg.del_proj(idx='all')

        _reset_eeg_montage(epochs_eeg)

        self.epochs_meg = epochs_meg
        self.epochs_eeg = epochs_eeg

        logger.debug(f'Got epochs_meg: {epochs_meg}'.replace('\n', ' '))
        logger.debug(f'Got epochs_eeg: {epochs_eeg}'.replace('\n', ' '))

        return epochs_meg, epochs_eeg


# %% ---- 2024-03-27 ------------------------
# Play ground


# %% ---- 2024-03-27 ------------------------
# Pending


# %% ---- 2024-03-27 ------------------------
# Pending
