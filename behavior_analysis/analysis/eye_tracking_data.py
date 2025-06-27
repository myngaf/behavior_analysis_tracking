from behavior_analysis.experiment import  BehaviorExperiment
from ..utilities import print_heading, print_subheading, manage_files_helpers
from ..utilities.plotting_helpers import BasePlotting
from .csv import import_csvs
import numpy as np
import pandas as pd
import os

from pathlib import Path
from typing import Union

from sklearn.neighbors import KernelDensity


class EyeConvergenceAnalyser(BasePlotting):

    def __init__(self, data, bandwidth=2.0, default_threshold=50., threshold_limits=(35, 65), verbose=True, **kwargs):
        BasePlotting.__init__(self, **kwargs)
        self.data = data
        self.bandwidth = bandwidth
        self.verbose = verbose
        self.threshold = default_threshold
        self.min_threshold = min(threshold_limits)
        self.max_threshold = max(threshold_limits)

    @staticmethod
    def import_eye_angles(paths):
        """Import eye angle data from kinematic files.

        Parameters
        ----------
        paths : iterable
            List of paths to files containing kinematic data from tracked video

        Returns
        -------
        fish_angles : pd.DataFrame
            DataFrame containing left eye angle, right eye angle and eye convergence angle across all frames
        """
        print('Importing data ({} files)'.format(len(paths)))
        fish_angles = []
        for i, path in enumerate(paths):
            print(i + 1,)
            kinematics = pd.read_csv(path)
            tracked = kinematics[kinematics['tracked']]
            eye_angles = tracked[['right_angle', 'left_angle']].applymap(np.degrees)
            fish_angles.append(eye_angles)
        fish_angles = pd.concat(fish_angles, ignore_index=True, axis=0)
        fish_angles['convergence'] = (fish_angles['right_angle'] - fish_angles['left_angle'])
        print('\ndone!\n')
        return fish_angles

class EyeTrackingData(object):
    """Class for analysing eye angle data from multiple animals.

    Parameters
    ----------
    data : dict or iterable
        If dict, should be key, value pairs where keys are unique identifiers for animals in an experiment and values
        are DataFrames containing left, right and convergence angle data. If iterable, should be DataFrames with unique
        identifiers passed to IDs argument.

    IDs : iterable (optional)
        List of unique identifiers for DataFrames passed to data argument.

    metadata : pd.DataFrame (optional)
        DataFrame containing metadata for each animal in the experiment."""

    # def __init__(self, data: pd.DataFrame, metadata: pd.DataFrame = None):
    #     super().__init__(data)
    #     assert isinstance(self.data.index, pd.MultiIndex), 'Data must be MultiIndexed!'
    #     self.metadata = metadata

    def __init__(self, data, IDs=None, metadata=None):
        if isinstance(data, dict):
            self.data = data
        elif IDs is not None:
            self.data = dict(zip(IDs, data))
        else:
            raise ValueError('Invalid data type')
        self.metadata = metadata
        self.analysers = {}

    @classmethod
    def from_experiment(cls, path):
        """Imports eye angle data from an experiment.

        Parameters
        ----------
        path : str or experiment.TrackingExperiment
            Complete path to the experiment directory or a TrackingExperiment object.
        """
        print_heading('IMPORTING EYE ANGLE DATA')
        if isinstance(path, str):
            experiment = BehaviorExperiment.open(path)
            experiment.open()
        else:
            experiment = path
        eye_data = {}
        for idx, fish_info in experiment.animal_data.iterrows():
            print_subheading(fish_info.ID)
            kinematics_directory = experiment.subdirs["kinematics"].joinpath(fish_info.ID)
            kinematics_files, kinematics_paths = manage_files_helpers.get_files(kinematics_directory)
            # fish_angles = EyeConvergenceAnalyser.import_eye_angles(*kinematics_paths)
            fish_angles = EyeConvergenceAnalyser.import_eye_angles(kinematics_paths)
            eye_data[fish_info.ID] = fish_angles
        metadata_columns = ['ID', 'date', 'name']
        if experiment.conditions:
            metadata_columns.append('condition')
        else:
            print("No conditions are set.")
        metadata = experiment.data[metadata_columns]
        return EyeTrackingData(eye_data, metadata=metadata)

    @property
    def concatenated_data(self):
        return pd.concat([df for df in self.data.itervalues()], ignore_index=True, axis=0)

    def calculate_convergence_scores(self, save_plots_to=None, **kwargs):
        """Calculates eye convergence score for each animal in an experiment.

        Parameters
        ----------
        save_plots_to : str (optional)
            Path to a directory where plots will be saved. If None, then plots are not saved.
        kwargs : dict
            Keyword arguments passed to EyeConvergenceAnalyser.

        Returns
        -------
        convergence_scores : pd.DataFrame
            DataFrame containing unique ID identifier for each animal in the experiment, the convergence score for the
            animal, and the threshold used to compute the convergence score. If metadata attribute is filled, this
            information is also included."""

        print_heading('CALCULATING CONVERGENCE SCORES')
        IDs, scores, thresholds = [], [], []
        for ID, eye_angles in self.data.iteritems():
            print_subheading(ID)

            ECA = EyeConvergenceAnalyser(eye_angles['convergence'], **kwargs)
            ECA.kernel_density_estimation()
            ECA.find_convergence_threshold()
            ECA.calculate_convergence_score()
            print('')

            IDs.append(ID)
            scores.append(ECA.convergence_score)
            thresholds.append(ECA.threshold)

            if save_plots_to is not None:
                plot_path = os.path.join(save_plots_to, ID + '.png')
                ECA.plot_threshold(save=True, output_path=plot_path)

            self.analysers[ID] = ECA

        convergence_scores = pd.DataFrame(zip(IDs, scores, thresholds), columns=['ID', 'score', 'threshold'])
        if self.metadata is not None:
            convergence_scores = self.metadata.join(convergence_scores.set_index('ID'), on='ID')
        return convergence_scores

    # def calculate_convergence_distribution(self, **kwargs):
    #     """Calculates the histogram of eye convergence angles across all animals.

    #     Parameters
    #     ----------
    #     kwargs : dict
    #         Keyword arguments passed to EyeConvergenceAnalyser.

    #     Returns
    #     -------
    #     eye_convergence_counts : np.ndarray
    #         (N x 3) array with N-1 bins. First column containing bin edges, second column contains normalised counts in
    #         each bin, third column contains values from kernel density estimation.
    #     """
    #     print_heading('CALCULATING CONVERGENCE DISTRIBUTION')
    #     all_convergence_angles = self.concatenated_data['convergence']
    #     ECA = EyeConvergenceAnalyser(all_convergence_angles, **kwargs)
    #     ECA.kernel_density_estimation()
    #     counts = np.concatenate([ECA.counts, [np.nan]])
    #     assert len(ECA.bin_edges) == len(counts) == len(ECA.kde_counts)
    #     eye_convergence_counts = np.array([ECA.bin_edges, counts, ECA.kde_counts])
    #     print('')
    #     return eye_convergence_counts

    # def calculate_angle_distribution(self):
    #     """Calculates a 2D histogram of eye angles across all animals.

    #     Returns
    #     -------
    #     eye_angle_counts : np.ndarray
    #         (3 x 45 x 45) arrays. Axis 0 encodes left eye bin values, right eye bin values and the 2D kernel density
    #         estimation of the probability of the corresponding left-right angle pair.
    #     """
    #     print_heading('CALCULATING ANGLE DISTRIBUTION')
    #     eye_angles = self.concatenated_data[['left', 'right']].values
    #     eye_angles[:, 0] *= -1
    #     x, y = np.meshgrid(np.arange(0, 45), np.arange(0, 45))
    #     xy = np.vstack((x.ravel(), y.ravel())).T
    #     eye_kde = KernelDensity(1).fit(eye_angles)
    #     eye_log_density = eye_kde.score_samples(xy)
    #     eye_density_map = np.exp(eye_log_density).reshape((45, 45))
    #     eye_angle_counts = np.array([x, y, eye_density_map])
    #     print('')
    #     return eye_angle_counts
