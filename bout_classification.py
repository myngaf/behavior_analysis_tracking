from behavior_analysis.experiment import BehaviorExperiment
from behavior_analysis.analysis.bouts import BoutData
from behavior_analysis.analysis.eye_convergence import calculate_convergence
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform
from pathlib import Path
import numpy as np
from behavior_analysis.analysis.bout_mapping import calculate_distance_matrix_templates, interpolate_nd
from matplotlib import pyplot as plt


if __name__ == "__main__":

    experiment = BehaviorExperiment.open(r"J:\Martin Schneider\MS007")  # r"D:\DATA\behaviour\test"
    print(experiment)
    bouts_path = experiment.subdirs['analysis'].joinpath('bouts.csv')
    bouts_df = pd.read_csv(bouts_path, dtype={'ID': str, 'video_code': str})
    video_info = pd.read_csv(experiment.directory.joinpath('video_data.csv'), dtype={'ID': str, 'video_code': str})

    # Average eye convergence over 20 ms
    window = 0.02

    # Import convergence data
    fish_convergence = pd.read_csv('', dtype={'ID': str})
    convergence_states = []
    # Import bout data
    bouts = BoutData.from_metadata(bouts_df, experiment.subdirs['kinematics'], tail_only=False)
    for i, bout in bouts.iter():
        # Bout info
        bout_info = bouts.metadata.loc[i]
        fps = video_info[video_info["code"] == bout_info.code].squeeze().fps
        ID = bout_info.ID
        fish_info = fish_convergence[fish_convergence["ID"] == ID].squeeze()
        # Calculate convergence
        bout_convergence = np.degrees(calculate_convergence(bout))
        w = int(window * fps)
        convergence_start = bout_convergence[:w].mean()
        convergence_end = bout_convergence[-w:].mean()
        convergence_states.append(np.array([convergence_start, convergence_end]) >= fish_info.threshold)
    convergence_states = np.array(convergence_states)
    # Find bout phases
    spontaneous = (~convergence_states[:, 0]) & (~convergence_states[:, 1])
    early = (~convergence_states[:, 0]) & (convergence_states[:, 1])
    mid = (convergence_states[:, 0]) & (convergence_states[:, 1])
    late = (convergence_states[:, 0]) & (~convergence_states[:, 1])
    phase_labels = np.column_stack([spontaneous, early, mid, late])
    phase_labels = np.argwhere(phase_labels)[:, 1]
    bouts_df['phase'] = phase_labels
    bouts_df.to_csv(experiment.subdirs["analysis"].joinpath("bout_convergence_labels.csv"), index=False)
