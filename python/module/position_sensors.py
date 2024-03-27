"""
File: analysis.py
Author: Chuncheng Zhang
Date: 2024-03-27
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Analysis on epochs

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-03-27 ------------------------
# Requirements and constants
import mne
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from scipy.spatial.distance import cdist

from pathlib import Path

from .load_data import JointData
from . import logger


# %% ---- 2024-03-27 ------------------------
# Function and class

# %% ---- 2024-03-27 ------------------------
# Play ground


def apply_trans(trans, pts, move=True):
    """Apply a transform matrix to an array of points.

    Parameters
    ----------
    trans : array, shape = (4, 4) | instance of Transform
        Transform matrix.
    pts : array, shape = (3,) | (n, 3)
        Array with coordinates for one or n points.
    move : bool
        If True (default), apply translation.

    Returns
    -------
    transformed_pts : shape = (3,) | (n, 3)
        Transformed point(s).
    """

    if isinstance(trans, dict):
        trans = trans['trans']
    pts = np.array(pts)
    if pts.size == 0:
        return pts.copy()

    # apply rotation & scale
    out_pts = np.dot(pts, trans[:3, :3].T)
    # apply translation
    if move:
        out_pts += trans[:3, 3]

    return out_pts


def sensor_positions(epochs: mne.Epochs):
    info = epochs.info
    dev_head_t = info['dev_head_t']
    df = pd.DataFrame(
        [
            (e['ch_name'], str(e['kind']), str(e['unit']), e['loc'][:3])
            for e in info['chs']
        ],
        columns=['ch_name', 'kind', 'uint', 'loc']
    )
    df['pos'] = df['loc'].map(lambda e: apply_trans(dev_head_t, e[:3]))
    logger.debug(f'Found chs:\n{df}')
    return df


class JointData_PositionSensors(JointData):
    meg_sensor_positions = None
    eeg_sensor_positions = None
    distance_matrix = None

    def __init__(self, path: Path):
        super().__init__(path)
        self.position_sensors()
        pass

    def position_sensors(self):
        meg = sensor_positions(self.epochs_meg)
        eeg = sensor_positions(self.epochs_eeg)

        m_pos = np.array(meg['pos'].to_list())
        e_pos = np.array(eeg['pos'].to_list())
        c = cdist(e_pos, m_pos)
        dm = pd.DataFrame(c, index=eeg['ch_name'], columns=meg['ch_name'])
        logger.debug(f'Distance matrix:\n{dm}')

        self.meg_sensor_positions = meg
        self.eeg_sensor_positions = eeg
        self.distance_matrix = dm

    def plot_positions(self) -> plt.Figure:
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        kwargs = dict(
            ch_groups='position',
            show=False,
            linewidth=1,
        )

        # --------------------
        ax = axs[0, 0]
        ax.set_title('MEG sensors')
        mne.viz.plot_sensors(self.epochs_meg.info, title='MEG',
                             axes=ax, show_names=False, **kwargs)

        # --------------------
        ax = axs[0, 1]
        ax.set_title('EEG sensors')
        mne.viz.plot_sensors(self.epochs_eeg.info, title='EEG',
                             axes=ax, show_names=True, **kwargs)

        # --------------------
        gs = axs[1, 0].get_gridspec()
        axs[1, 0].remove()
        axs[1, 1].remove()
        ax = fig.add_subplot(gs[1:, 0:])
        ax.set_title('Distance')
        sns.heatmap(self.distance_matrix, ax=ax)

        # --------------------
        ax = axs[1][1]
        ax.axis('off')

        fig.suptitle('Sensor positions')
        fig.tight_layout()
        return fig

    def plot_positions_3d(self):
        df = pd.concat([self.meg_sensor_positions,
                       self.eeg_sensor_positions]).copy()
        df['x'] = df['pos'].map(lambda e: e[0])
        df['y'] = df['pos'].map(lambda e: e[1])
        df['z'] = df['pos'].map(lambda e: e[2])
        df['size'] = 1
        fig = px.scatter_3d(df, x='x', y='y', z='z',
                            size='size', size_max=10, color='kind')
        return fig


# %% ---- 2024-03-27 ------------------------
# Pending


# %% ---- 2024-03-27 ------------------------
# Pending
