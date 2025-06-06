from behavior_analysis.experiment import BehaviorExperiment
from behavior_analysis.analysis.bouts import BoutData
from behavior_analysis.analysis.bout_mapping import calculate_distance_matrix_templates, interpolate_nd
from behavior_analysis.utilities.timer import Timer
import pandas as pd
import numpy as np
from pathlib import Path


if __name__ == "__main__":
    # Open template bouts
    template_directory = Path(r'J:\Duncan Mearns\behavior_mapping')
    template_frame_rate = 500.
    # Open 1744 exemplar bouts representative of all behaviors
    exemplars_df = pd.read_csv(template_directory.joinpath('exemplars.csv'),
                               dtype={'ID': str, 'code': str})
    # Open tail statistics and eigenfish for bout mapping
    eigenfish = np.load(template_directory.joinpath('eigenfish.npy'))
    eigenfish = eigenfish[:3]  # take first three eigenfish only
    mean, std = np.load(template_directory.joinpath('tail_statistics.npy'))
    # Import template bouts
    templates = BoutData.from_metadata(exemplars_df, template_directory.joinpath("kinematics"))
    # Map template bouts onto eigenfish
    templates = templates.map(vectors=eigenfish, whiten=True, mean=mean, std=std)
    templates = templates.to_list(values=True)
    print(len(templates))
    print(templates[0].shape)

    # Open experiment
    experiment = BehaviorExperiment.open(r"D:\DATA\behaviour\test")
    print(experiment)
    # Open video and bout info
    video_info = pd.read_csv(experiment.directory.joinpath('video_data.csv'), dtype={'ID': str, 'code': str})
    bouts_df = pd.read_csv(experiment.subdirs["analysis"].joinpath('bouts.csv'),
                           dtype={'ID': str, 'code': str})
    # Import bouts
    bouts = BoutData.from_metadata(bouts_df, experiment.subdirs["kinematics"], tail_only=True)
    # Map bouts onto eigenfish
    bouts = bouts.map(vectors=eigenfish, whiten=True, mean=mean, std=std)
    # Start timer
    timer = Timer()
    analysis_times = []
    timer.start()
    # Iterate through fish
    output_directory = experiment.subdirs["analysis"].joinpath('bout_distances')
    if not output_directory.exists():
        output_directory.mkdir(parents=True)
    for ID in bouts.metadata['ID'].unique():
        # Save distance matrix for each fish to bout_distances folder in analysis directory
        output_path = output_directory.joinpath(ID + '.npy')
        if not output_path.exists():
            print(ID + '...', end=' ')
            # Interpolate bouts to correct frame rate
            fish_bouts = []
            for i, bout in bouts.iter(IDs=[ID], values=True):
                code = bouts.metadata.loc[i, 'code']
                fps = video_info[video_info['code'] == code].squeeze().fps
                interp = interpolate_nd(bout, fps, template_frame_rate)
                fish_bouts.append(interp)
            # Calculate distance matrix
            D = calculate_distance_matrix_templates(fish_bouts, templates, fs=template_frame_rate)
            # Save distance matrix
            np.save(output_path, D)
            # Show time taken
            time_taken = timer.lap()
            analysis_times.append(time_taken)
            print(timer.convert_time(time_taken))
    average_time = timer.average
    print(f'Total time: {timer.convert_time(timer.stop())}')
    print(f'Average time: {timer.convert_time(average_time)}')
