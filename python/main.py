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
import mne
import matplotlib.pyplot as plt

from module import logger
from module.files import search_all_files
from module.load_data import JointData
from module.position_sensors import JointData_PositionSensors


# %% ---- 2024-03-27 ------------------------
# Function and class


# %% ---- 2024-03-27 ------------------------
# Play ground
if __name__ == '__main__':
    logger.info('Started')
    files_table = search_all_files()
    print(f'Found files: {files_table}')

    path = files_table.query('experiment == "complex"').iloc[0]['path']
    jd = JointData_PositionSensors(path)

    fig = jd.plot_positions()
    plt.show()

    fig = jd.plot_positions()
    # fig = jd.plot_positions_3d()
    fig.show()


# %% ---- 2024-03-27 ------------------------
# Pending

# %% ---- 2024-03-27 ------------------------
# Pending


# %%
