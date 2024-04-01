"""
File: match_montage.py
Author: Chuncheng Zhang
Date: 2024-03-29
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Match the existing montage with the other montage

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-03-29 ------------------------
# Requirements and constants
import mne
import numpy as np
from .constants import FIFF


# %% ---- 2024-03-29 ------------------------
# Function and class

def compute_trans(src_montage, dst_montage):
    # Compute trans, it makes
    # src x trans = dst
    src = np.eye(4)
    dst = np.eye(4)

    src[:3, :3] = np.array(
        [e['r'] for e in src_montage.dig if e['kind'] == FIFF.FIFFV_POINT_CARDINAL])
    dst[:3, :3] = np.array(
        [e['r'] for e in dst_montage.dig if e['kind'] == FIFF.FIFFV_POINT_CARDINAL])

    return np.eye(4)

    return np.matmul(np.linalg.inv(src), dst)


class MontageMatcher(object):
    '''
    The montage is existing montage,
    and the standard_montage contains the sensors' position
    which I require,
    the aim is match the standard_montage to the existing montage.

    The pipelines are:
        1. affine transform standard_montage to montage (stay unchanged)
        2. set montage digs' position the standard_montage digs' position (stay unchanged)
    '''
    standard_montage = None
    montage = None
    trans = None

    eeg_ch_table = {}

    def __init__(self, montage: mne.channels.DigMontage, standard_montage: mne.channels.DigMontage):
        self.montage = montage
        self.standard_montage = standard_montage
        # Prepare trans
        self.compute_trans()
        # Pipeline: 1
        self.transform()

    def compute_trans(self):
        # Prepare trans
        self.trans = compute_trans(self.standard_montage, self.montage)

    def transform(self):
        # Pipeline: 1
        for e in self.standard_montage.dig:
            e['r'] = self._transform(e['r'])

        names = self.standard_montage.ch_names
        digs = [e for e in self.standard_montage.dig
                if e['kind'] == FIFF.FIFFV_POINT_EEG]

        for name, dig in zip(names, digs, strict=True):
            self.eeg_ch_table[name.lower()] = dict(name=name, dig=dig)

    def _transform(self, vec: np.ndarray):
        v = np.zeros((1, 4))
        v[0, :3] = vec
        dst = np.matmul(v, self.trans)[0, :3].ravel()
        print(vec, '->', dst)
        return dst


# %% ---- 2024-03-29 ------------------------
# Play ground


# %% ---- 2024-03-29 ------------------------
# Pending


# %% ---- 2024-03-29 ------------------------
# Pending
