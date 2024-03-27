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

from pathlib import Path

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
    channels_in_order = [_ch_name(e)
                         for e in channels_in_order_5x7.split(',')]
    montage64 = mne.channels.make_standard_montage(montage64_name)

    for name in channels_in_order:
        assert name in montage64.ch_names, f'Unknown channel: {name}'
    assert len(
        epochs.ch_names) == 35, f'Incorrect number of channels, {epochs}'

    rename = {}
    for j in range(35):
        rename[epochs.ch_names[j]] = channels_in_order[j]
    epochs.rename_channels(rename)

    epochs.set_montage(montage64, on_missing='warn')

    return epochs


class JointData(object):
    path = None
    raw = None
    events = None
    event_id = None
    epochs_meg = None
    epochs_eeg = None

    experiment_events = {
        '1': '手',
        '2': '腕',
        '3': '肘',
        '4': '肩',
        '5': '静息',
    }

    def __init__(self, path: Path):
        self.load_raw(path)
        self.get_epochs()

    def load_raw(self, path: Path):
        raw = mne.io.read_raw_ctf(path)
        events, event_id = mne.events_from_annotations(raw)

        self.path = path
        self.raw = raw
        self.events = events
        self.event_id = event_id

        logger.debug(f'Loaded raw: {path}, {raw}')

        return raw, events, event_id

    def get_epochs(self):
        raw = self.raw
        events = self.events
        event_id = self.event_id

        epochs_meg = mne.Epochs(raw, events, event_id, picks='mag')

        picks = [e for e in raw.ch_names if e.startswith('EEG')][:35]
        epochs_eeg = mne.Epochs(raw, events, event_id, picks=picks)

        _reset_eeg_montage(epochs_eeg)

        self.epochs_meg = epochs_meg
        self.epochs_eeg = epochs_eeg

        logger.debug(f'Got epochs_meg: {epochs_meg}, epochs_eeg: {epochs_eeg}')

        return epochs_meg, epochs_eeg


# %% ---- 2024-03-27 ------------------------
# Play ground


# %% ---- 2024-03-27 ------------------------
# Pending


# %% ---- 2024-03-27 ------------------------
# Pending
