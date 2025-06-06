from behavior_analysis.experiment import BehaviorExperiment
from behavior_analysis.analysis.bouts import BoutData
from behavior_analysis.analysis.bout_mapping import calculate_distance_matrix
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform
from pathlib import Path
import numpy as np
from behavior_analysis.analysis.bout_mapping import calculate_distance_matrix_templates, interpolate_nd
from matplotlib import pyplot as plt


if __name__ == "__main__":

    template_directory = Path(r'J:\Duncan Mearns\behavior_mapping')
    template_frame_rate = 500.
    eigenfish = np.load(template_directory.joinpath('eigenfish.npy'))
    eigenfish = eigenfish[:3]  # take first three eigenfish only
    mean, std = np.load(template_directory.joinpath('tail_statistics.npy'))

    # Create experiment
    # experiment = BehaviorExperiment.create(r"D:\DATA\behaviour\test")
    # Open experiment
    experiment = BehaviorExperiment.open(r"J:\Martin Schneider\MS007")  #   r"D:\DATA\behaviour\test"
    print(experiment)
    # # Update animal info
    # experiment.update()
    # # Calculate backgrounds
    # experiment.calculate_backgrounds(parallel=True, n_processes=4)
    # # Set tracking thresholds
    # experiment.set_thresholds()
    # # Run tracking
    # experiment.track()
    # # Run kinematics
    # experiment.kinematics(parallel=True, n_processes=4)
    # experiment.find_bouts(check_thresholds=False)

    video_info = pd.read_csv(experiment.directory.joinpath('video_data.csv'), dtype={'ID': str, 'code': str})
    bouts_df = pd.read_csv(experiment.subdirs["analysis"].joinpath('bouts.csv'), dtype={'ID': str, 'code': str})
    codes = bouts_df["code"].unique()
    keep = [codes[0], codes[1], codes[-2], codes[-1]]
    bouts_df = bouts_df[bouts_df['code'].isin(keep)]
    # print(bouts_df)
    bouts = BoutData.from_metadata(bouts_df, experiment.subdirs["kinematics"], tail_only=True)
    bouts = bouts.map(vectors=eigenfish, whiten=True, mean=mean, std=std)

    for ID in bouts.metadata['ID'].unique():
        print(bouts.get(IDs=[ID]).data)
        # for i, bout in bouts.iter(IDs=[ID], values=True):
        #     code = bouts.metadata.loc[i, 'code']
        #     fps = video_info[video_info['code'] == code].squeeze().fps
        #     interp = interpolate_nd(bout, fps, template_frame_rate)

        #     plt.plot(interp[:, 0], interp[:, 1])
        #     plt.show()
        #     break
        # break

    # print(len(mapped))
    # print(mapped[0])

    # D = calculate_distance_matrix(mapped)
    # D = squareform(D)
    # plt.matshow(D)
    # plt.show()
