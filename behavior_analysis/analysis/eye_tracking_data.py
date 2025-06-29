from behavior_analysis.experiment import  BehaviorExperiment
from ..utilities import print_heading, print_subheading, manage_files_helpers
from ..utilities.plotting_helpers import BasePlotting
from .csv import import_csvs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    def kernel_density_estimation(self):
        """Performs kernel density estimation of the data using a gaussian with the given bandwidth.

        Returns
        -------
        self

        Notes
        -----
        After calling this method, the following attributes become accessible:
        min_angle, max_angle, bin_edges, counts, kde
        """
        self.min_angle = np.floor(self.data.min())
        self.max_angle = np.ceil(self.data.max())
        self.bin_edges = np.arange(self.min_angle, self.max_angle + 1)
        self.counts, self.bin_edges = np.histogram(self.data, bins=self.bin_edges)
        if self.verbose:
            print('Performing kernel density estimation...',)
        # perform kernel density estimation
        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(np.expand_dims(self.data, 1))
        # get the log counts
        log_counts = self.kde.score_samples(np.expand_dims(self.bin_edges, 1))
        # convert logarithmic values to absolute counts
        self.kde_counts = np.exp(log_counts)
        # find the value of the mode
        self.mode = self.bin_edges[np.argmax(self.kde_counts)]
        if self.verbose:
            print('done!')
        return self

    def find_convergence_threshold(self):
        """Finds the antimode of kernel density estimated distribution and sets it as the convergence threshold.

        Returns
        -------
        threshold : int
            The threshold used to define eye convergence (the antimode of the distribution of eye vergence angles).

        Notes
        -----
        This method finds the local minimum of the eye convergence distribution that lies within the threshold limits.
        If a local minimum does not exist within the limit, the convergence threshold defaults to the initial value.

        Smoothed distribution:
           _
          / \   _
         /   \_/ \
        /         \

        First derivative:

        + \     _
        0  \   / \
        -   \_/   \

        Local maxima and minima occur when the first derivative is zero. Specifically, local minima occur when there is
        an inversion of the sign of the first derivative (from negative to positive).

        Sign of first derivative:

        + .    .
        0  .  . ..
        -   ..    .

        Second derivative:

         (local min)
              |
              V
        +     _
        0    / \_
        -  _/    \

        """
        # take the derivative of the distribution
        diffed = np.diff(self.kde_counts)
        # smooth the differentiated data
        smoothed = pd.Series(diffed).rolling(7, min_periods=0, center=True).mean().values
        # take the sign of the smoothed data (is the function increasing or decreasing)
        signed = np.sign(smoothed)
        # take the derivative of the sign of the first derivative
        second_diff = np.diff(signed)
        # find the indices of local minima (i.e. where the sign of first derivative goes from negative to positive)
        local_minima = np.where(second_diff > 0)[0] + 1
        # find values of the antimodes
        antimodes = self.bin_edges[local_minima]
        try:
            # Try to find an antimode within the threshold range
            self.threshold = antimodes[(antimodes > self.min_threshold) & (antimodes < self.max_threshold)][0]
        except IndexError:  # local minimum does not exist within the threshold range
            if self.verbose:
                print('No local minimum within limits!')
        if self.verbose:
            print('Eye convergence threshold:', self.threshold)
        return self.threshold

    def calculate_convergence_score(self):
        """Calculates the convergence score, i.e. the proportion of frames above the convergence threshold.

        Returns
        -------
        convergence_score: float
            The proportion of frames that are above the convergence threshold.
        """
        # find frames that are above threshold
        above_threshold = self.data[self.data >= self.threshold]
        # get number of frames above threshold
        converged_counts = len(above_threshold)
        # get total number of frames in the data
        total_counts = len(self.data)
        # calculate proportion of frames that are above threshold
        score = float(converged_counts) / float(total_counts)
        self.convergence_score = score
        if self.verbose:
            print('Eye convergence score:', self.convergence_score)
        return self.convergence_score

    def plot_histogram(self, save=False, output_path=None):
        """Plots a histogram of eye vergence angles.

        Parameters
        ----------
        save : bool, optional (default = False)
            - True: the figure is saved to the specified or default output path
            - False: the figure is shown in a window

        output_path : str, optional (default = None)
            Where to save the figure if save == True. The default path is: 'histogram_of_eye_convergence_angles.png'
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle('Distribution of eye vergence angles')

        ax.hist(self.data, bins=self.bin_edges)

        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Number of frames')

        if save:
            if output_path is None:
                output_path = 'histogram_of_eye_convergence_angles.png'
            self.save_figure(fig, output_path)
        else:
            plt.show()

    def plot_kernel_density_estimation(self, save=False, output_path=None):
        """Plots a histogram of observed eye vergence angles and the estimated distribution.

        Parameters
        ----------
        save : bool, optional (default = False)
            - True: the figure is saved to the specified or default output path
            - False: the figure is shown in a window

        output_path : str, optional (default = None)
            Where to save the figure if save == True. The default path is:
            'kernel_density_estimation_of_eye_convergence_angles.png'
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle('Kernel density estimation of eye vergence angles')

        ax.hist(self.data, bins=self.bin_edges)
        ax.plot(self.bin_edges, self.kde_counts * len(self.data), linewidth=3)

        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Number of frames')

        if save:
            if output_path is None:
                output_path = 'kernel_density_estimation_of_eye_convergence_angles.png'
            self.save_figure(fig, output_path)
        else:
            plt.show()

    def plot_threshold(self, save=False, output_path=None):
        """Plots a histogram of observed eye vergence angles and the estimated distribution.

        Makes two subplots. The left subplot shows the estimated distribution of eye vergence angles with everything
        above threshold shaded. The right subplot shows the observed distribution of eye vergence angles and the
        threshold (which is used to calculate the convergence_score).

        Parameters
        ----------
        save : bool, optional (default = False)
            - True: the figure is saved to the specified or default output path
            - False: the figure is shown in a window

        output_path : str, optional (default = None)
            Where to save the figure if save == True. The default path is: 'eye_convergence_threshold.png'
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
        fig.suptitle('Eye convergence threshold')

        converged = self.bin_edges >= self.threshold

        ax1.plot(self.bin_edges, self.kde_counts * len(self.data), linewidth=3)
        ax1.fill_between(self.bin_edges[converged], 0, self.kde_counts[converged] * len(self.data))

        ax1.set_title('Kernel density estimation')
        ax1.set_xlabel('Angle (degrees)')
        ax1.set_ylabel('Counts')

        ax2.hist(self.data, bins=self.bin_edges)
        ax2.plot([self.threshold, self.threshold], [0, self.counts.max()], c='k', linestyle='dashed')

        ax2.set_title('Raw counts')
        ax2.set_xlabel('Angle (degrees)')

        ax1.set_xlim(-25, 100)
        ax1.set_xticks(np.arange(-25, 125, 25))

        if save:
            if output_path is None:
                output_path = 'eye_convergence_threshold.png'
            self.save_figure(fig, output_path)
            plt.close(fig)
        else:
            plt.show()

    plot_kde = plot_kernel_density_estimation

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
        path : str or experiment (BehaviorExperiment)
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
            if 'condition' in experiment.animal_data.columns:
                metadata_columns.append('condition')
            else:
                print("No conditions are set.")
            metadata = experiment.animal_data[metadata_columns]
        return EyeTrackingData(eye_data, metadata=metadata)

        # original code:
        # metadata_columns = ['ID', 'date', 'name']
        # if experiment.conditions:
        #     metadata_columns.append('condition')
        # metadata = experiment.data[metadata_columns]
        # return EyeTrackingData(eye_data, metadata=metadata)

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
        for ID in self.metadata["ID"]: #need to fix this!!!
            print_subheading(ID)

            ECA = EyeConvergenceAnalyser(self.data[ID]['convergence'], **kwargs)
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
