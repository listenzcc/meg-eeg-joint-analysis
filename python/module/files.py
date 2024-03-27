"""
File: files.py
Author: Chuncheng Zhang
Date: 2024-03-27
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Search for the use-able files

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-03-27 ------------------------
# Requirements and constants
import os
import pandas as pd

from pathlib import Path

from . import logger

data_path = Path(os.environ['HOME']+'/nfsHome/meg-eeg-joint-analysis/data')

# %% ---- 2024-03-27 ------------------------
# Function and class


def _check_known_file(folder_full_path: str) -> bool:
    # Parse input
    path = Path(folder_full_path)
    name = path.name

    # Check if it is legal
    checks = [
        name.startswith('S'),
        name.startswith('Noise'),
        name.endswith('.ds')
    ]
    all_good_flag = (checks[0] or checks[1]) and checks[2]
    assert all_good_flag, 'Invalid folder name'
    # logger.debug(f'Found known folder: {folder_full_path}')

    # Parse all
    return dict(
        subject_id=path.parent.name,
        session_id=name,
        session_type='noise' if name.startswith('Noise') else 'experiment',
        path=path
    )


def _mark_experiments(raw_df: pd.DataFrame):
    '''
    Mark the experiment with simple and complex experiment.
    '''
    df = raw_df.copy()

    # Group by session_type
    group = df.query('session_type=="experiment"').groupby('subject_id')
    logger.debug(f'Found experiment sessions:\n{group.count()}')

    # Mark the tail 8 sessions as the **complex**
    # and the other sessions as the **simple**
    # if the session_type is "noise", mark them as the **noise**
    df1 = group.tail(8)
    logger.debug(f'Complex experiment sessions are:\n{df1.count()}')

    df['experiment'] = df['session_type'].map(
        lambda e: 'simple' if e == 'experiment' else e)
    df.loc[df1.index, 'experiment'] = 'complex'

    return df


def search_all_files(path: Path = data_path):
    found = []
    for folder_full_path, _, _ in os.walk(path):
        try:
            found.append(_check_known_file(folder_full_path))
        except AssertionError as e:
            pass
    found = sorted(found, key=lambda e: e['session_id'])
    df = pd.DataFrame(found)
    return _mark_experiments(df)


# %% ---- 2024-03-27 ------------------------
# Play ground


# %% ---- 2024-03-27 ------------------------
# Pending


# %% ---- 2024-03-27 ------------------------
# Pending
