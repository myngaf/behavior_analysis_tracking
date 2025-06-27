import datetime
import json
import sys
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import pandas as pd

from behavior_analysis.gui.applications import SetThresholdsApp, BoutDetectionThresholdApp, SetConditionsApp
from behavior_analysis.gui import SetCircularMask
from .naming import generate_ID, generate_video_code
from ..tracking import MaskGenerator
from ..tracking import Tracker, Kinematics
from ..tracking.background import BackgroundGenerator
from ..tracking.display import TrackedVideo
from ..tracking.segmentation import BoutDetector
from ..utilities import Timer, print_subheading, print_heading


# from video_analysis_toolbox.gui.apps import SetThresholdsApp

class BehaviorExperiment:

    animal_columns = ('ID',
                      'date',
                      'name',
                      'video_directory',
                      'background',
                      'tracking',
                      'kinematics',
                      'thresh1',
                      'thresh2',
                      'blur',
                      'dp',
                      'mindist',
                      'param1',
                      'param2')
    animal_data_types = {'ID': str}
    ID_function = staticmethod(generate_ID)

    video_columns = ('code',
                     'path',
                     'ID',
                     'width',
                     'height',
                     'fps',
                     'length',
                     'tracked')
    video_data_types = {'code': str, 'ID': str, 'tracked': bool}
    video_code_function = staticmethod(generate_video_code)

    mask_columns = ('ID',
                    'center_x',
                    'center_y',
                    'radius')
    mask_data_types = {'ID': str}


    def __init__(self, directory, video_directory=None):

        # Experiment directory
        self.directory = Path(directory)
        subdirs = [path for path in self.directory.glob('*') if path.is_dir()]
        self.subdirs = dict([(path.stem, path) for path in subdirs])
        self.files = [path for path in self.directory.glob('*') if path.is_file()]

        # Video directory
        if video_directory is not None:
            self.video_directory = Path(video_directory)
        else:
            self.video_directory = self.directory.joinpath('videos')
        if not self.video_directory.exists():
            print(f'Video directory {self.video_directory} does not exist!')
            self.video_directory = None

        # Experiment metadata
        self.metadata_path = self.directory.joinpath('metadata.json')
        with open(self.metadata_path) as json_file:
            self.metadata = json.load(json_file)

        # Animal data
        self.animal_data_path = self.directory.joinpath(self.metadata['animal_data'])
        self.animal_data = pd.read_csv(self.animal_data_path, dtype=self.animal_data_types)

        # Video data
        self.video_data_path = self.directory.joinpath(self.metadata['video_data'])
        self.video_data = pd.read_csv(self.video_data_path, dtype=self.video_data_types)

        # Mask data
        self.mask_data_path = self.directory.joinpath(self.metadata['mask_data'])
        self.mask_data = pd.read_csv(self.mask_data_path, dtype=self.mask_data_types)

        metadata = zip([self.animal_data, self.video_data, self.mask_data],
                       [self.animal_columns, self.video_columns, self.mask_columns])
        for df, _list in metadata:
            for col in _list:
                if col not in df.columns:
                    df[col] = np.NaN

    @classmethod
    def create(cls, directory: Union[str, Path], video_directory: Union[str, Path] = None, **kwargs):
        """Creates a new experiment in the directory provided.

        Parameters
        ----------
        directory : str or Path
            Path to a directory.
        video_directory : str or Path (default = None)
            Path to a directory containing video files. If None, will check if a "videos" directory exists within the
            experiment directory. If not, methods that analyse videos will not be available.

        Other Parameters
        ----------------
        animal_data : str (default = 'fish_data.csv')
            The name of the data file that contains information about each animal in the experiment.
        video_data : str (default = 'video_data.csv')
            The name of the data file that contains information about each video in the experiment.
        **kwargs : dict
             Additional key, value pairs that are added to the metadata file for the experiment.
        """
        # Check directory path
        path = Path(directory)
        metadata_path = path.joinpath('metadata.json')
        if not path.exists():  # path does not exist
            create_new = cls.yes_no_question(f'The given directory {path} does not exist. '
                                             f'Create a new experiment in this directory?')
            if not create_new:
                print('Cannot create experiment. Exiting.')
                sys.exit()
            else:
                path.mkdir(parents=True)
        else:  # check if experiment already exists at this location
            if metadata_path.exists():
                with open(metadata_path) as json_file:
                    metadata = json.load(json_file)
                date_created = metadata['date']
                open_experiment = cls.yes_no_question(f'An experiment already exists at {path}, created on '
                                                      f'{date_created}. Open this experiment?')
                if open_experiment:
                    return cls.open(directory, video_directory=video_directory)
                else:
                    print('Cannot create experiment. Exiting.')
                    sys.exit()

        experiment_name = str(path.name)
        today = str(datetime.date.today())
        metadata = dict(name=experiment_name,
                        date=today,
                        animal_data=kwargs.get('animal_data', 'fish_data.csv'),
                        video_data=kwargs.get('video_data', 'video_data.csv'),
                        mask_data=kwargs.get('mask_data', 'mask_data.csv'))
        metadata.update(**kwargs)
        with open(metadata_path, 'w') as json_file:
            json.dump(metadata, json_file)

        animal_data = pd.DataFrame(columns=cls.animal_columns)
        animal_data.to_csv(path.joinpath(metadata['animal_data']), index=False)

        video_data = pd.DataFrame(columns=cls.video_columns)
        video_data.to_csv(path.joinpath(metadata['video_data']), index=False)

        mask_data = pd.DataFrame(columns=cls.mask_columns)
        mask_data.to_csv(path.joinpath(metadata['mask_data']), index=False)

        return cls(directory, video_directory=video_directory)

    @classmethod
    def open(cls, directory: Union[str, Path], video_directory: Union[str, Path] = None):
        """Opens a pre-existing experiment.

        Parameters
        ----------
        directory : str or Path
            Path to a pre-existing experiment directory.
        video_directory : str or Path (default = None)
            Path to a directory containing video files. If None, will check if a "videos" directory exists within the
            experiment directory. If not, methods that analyse videos will not be available.
        """
        # Check directory path
        path = Path(directory)
        metadata_path = path.joinpath('metadata.json')
        if not path.exists():
            print(f'No experiment exists at {path}. Exiting.')
            sys.exit()
        elif not metadata_path.exists():
            print(f'No metadata file exists for the experiment at {path}. Exiting.')
            sys.exit()
        return cls(directory, video_directory=video_directory)

    def __str__(self):
        return '\n'.join([f'{key}: {val}' for key, val in self.metadata.items()]) + '\n'

    @property
    def has_video_directory(self):
        """Used for internal checks whether a video directory exists."""
        return self.video_directory is not None

    @staticmethod
    def yes_no_question(q,
                        affirmative_answers=('y', 'yes', '1', 't', 'true'),
                        negative_answers=('n', 'no', '0', 'f', 'false')):
        """Asks the user a yes/no question in the command line.

        Parameters
        ----------
        q : str
            A question that the user can answer in the command line
        affirmative_answers : tuple
            Valid affirmative answers (case insensitive). Default: ('y', 'yes', '1', 't', 'true')
        negative_answers : tuple
            Valid negative answers (case insensitive). Default: ('n', 'no', '0', 'f', 'false')

        Returns
        -------
        bool : True for affirmative answers, False for negative answers

        Raises
        ------
        ValueError
            If unrecognised answer given
        """
        answer = input(q + ' [y/n] ')
        if answer.lower() in affirmative_answers:
            return True
        elif answer.lower() in negative_answers:
            return False
        else:
            raise ValueError(f'Invalid answer! Recognised responses: {affirmative_answers + negative_answers}')

    @staticmethod
    def convert_date(YYYY_MM_DD, delimiter='_'):
        """Converts a date string into a datetime object.

        Parameters
        ----------
        YYYY_MM_DD : str
            Date in the format (year, month, day) separated by delimiter
        delimiter : str, default '_'
            Delimiter between the year & month and month & day

        Returns
        -------
        date : datetime.date
            The date as a datetime.date object
        """
        digits = YYYY_MM_DD.split(delimiter)
        year, month, day = [int(dig) for dig in digits]
        date = datetime.date(year, month, day)
        return date

    def get(self, i=0):
        if isinstance(i, int):
            animal_data = self.animal_data.loc[i]
        elif isinstance(i, str):
            animal_data = self.animal_data[self.animal_data['ID'] == i].iloc[0]
        else:
            raise TypeError('i must be integer or string')
        animal_data = animal_data.to_dict()
        if self.has_video_directory:
            animal_data['video_directory'] = self.video_directory.joinpath(animal_data['video_directory'])
            videos = self.video_data[self.video_data['ID'] == animal_data['ID']]
            videos['path'] = videos['path'].apply(lambda path: self.video_directory.joinpath(path))
            animal_data['videos'] = videos
        else:
            animal_data['video_directory'] = None
            animal_data['videos'] = pd.DataFrame(columns=self.video_columns)
        return animal_data

    def iter(self, missing_data=None):
        if missing_data:
            data = self.animal_data[pd.isnull(self.animal_data[missing_data])]
        else:
            data = self.animal_data
        for i in data.index:
            yield i, self.get(i)

    def data_from_directory(self, directory):
        name = directory.name
        date_folder = directory.parent.name
        date = self.convert_date(date_folder)
        ID = self.ID_function(date_folder, name)
        relative_path = Path(date_folder).joinpath(name)
        if ID not in self.animal_data['ID'].values:
            # Information about animal
            animal_info = dict(zip(self.animal_columns, [None for col in self.animal_columns]))
            animal_info['ID'] = ID
            animal_info['date'] = date
            animal_info['name'] = name
            animal_info['video_directory'] = relative_path
            animal_info = pd.Series(animal_info, index=self.animal_columns)
            # Information about videos
            video_paths = directory.glob('*.avi')
            video_info = dict(zip(self.video_columns, [[] for col in self.video_columns]))
            for path in video_paths:
                video_info['code'].append(self.video_code_function(ID, str(path)))
                video_info['path'].append(relative_path.joinpath(path.name))
                video_info['ID'].append(animal_info['ID'])
                cap = cv2.VideoCapture(str(path))
                video_info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_info['fps'] = cap.get(cv2.CAP_PROP_FPS)
                video_info['length'] = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                video_info['tracked'].append(False)
            video_info = pd.DataFrame(video_info, columns=self.video_columns)
        else:
            animal_info = self.animal_data[self.animal_data['ID'] == ID].iloc[0]
            video_info = self.video_data[self.video_data['ID'] == ID]
        return animal_info, video_info

    def update(self):
        """Add new animals to the experiment."""
        if self.has_video_directory:
            for animal_directory in sorted(self.video_directory.glob('*/*')):
                animal_info, video_info = self.data_from_directory(animal_directory)
                if not animal_info['ID'] in self.animal_data['ID'].values:
                    self.animal_data = self.animal_data.append(animal_info, ignore_index=True)
                    self.video_data = self.video_data.append(video_info, ignore_index=True)
            self.animal_data.to_csv(self.animal_data_path, index=False)
            self.video_data.to_csv(self.video_data_path, index=False)

    def calculate_backgrounds(self, parallel=False, **kwargs):
        """Calculates background images for each animal in the experiment.

        Parameters
        ----------
        parallel : bool (default = False)
            Whether or not to compute the background image for each animal in parallel.

        Other Parameters
        ----------------
        n_processes : int
            Number of processes to run in parallel. Each animal runs in a separate process.

        See Also
        --------
        behavior_analysis.tracking.background.BackgroundGenerator
        """
        print_heading('CALCULATING BACKGROUNDS')
        background_generator = BackgroundGenerator('chunked', **kwargs)
        idxs, inputs, outputs = [], [], []
        for i, fish in self.iter(missing_data='background'):
            idxs.append(i)
            inputs.append(fish['videos']['path'].values)
            outputs.append(self.directory.joinpath('backgrounds', fish['ID'] + '.tiff'))
        if len(idxs):
            print(f'Calculating backgrounds for {len(idxs)} animals.')
            if parallel:
                background_generator.run_parallel(*list(zip(inputs, outputs)), verbose=True)
            else:
                for input_paths, output_path in zip(inputs, outputs):
                    print(output_path.stem + ':', end=' ')
                    _, t = background_generator.run(input_paths, output_path)
                    print(Timer.convert_time(t))
            self.animal_data.loc[idxs, 'background'] = True
            self.animal_data.to_csv(self.animal_data_path, index=False)
        print('Backgrounds up to date.')

    def set_circular_mask(self, **kwargs):
        # adjust blur in range 10-50 and param1 in range 10-100
        # press n for next fish and esc for quitting assignment of thresholds
        blur, dp, mindist, param1, param2 = 25, 1, 512, 50, 50
        for i, fish in self.iter(missing_data='blur'):
            print(f'Set mask thresholds for fish {fish["ID"]}')
            bg_path = self.directory.joinpath('backgrounds', fish['ID'] + '.tiff')
            masker = MaskGenerator(blur=blur, dp=dp, mindist=mindist, param1=param1, param2=param2)
            masker.read_img(bg_path)
            (blur, dp, mindist, param1, param2, key) = SetCircularMask(masker).show()
            if key == 27:
                break
            self.animal_data.loc[i, 'blur'] = blur
            self.animal_data.loc[i, 'dp'] = dp
            self.animal_data.loc[i, 'mindist'] = mindist
            self.animal_data.loc[i, 'param1'] = param1
            self.animal_data.loc[i, 'param2'] = param2
            self.animal_data.to_csv(self.animal_data_path, index=False)

    def calculate_circular_mask(self, **kwargs):
        print_heading('GENERATING CIRCULAR MASK')
        mask_info = dict(zip(self.mask_columns, [[] for col in self.mask_columns]))
        for i, fish in self.iter():
            # Check if thresholds exist
            print_subheading(fish['ID'])
            try:
                assert not pd.isnull(fish['blur'])
            except AssertionError:
                print(f'Thresholds not set for {fish["ID"]}.')
                continue
            # Assign path
            bg_path = self.directory.joinpath('backgrounds', fish['ID'] + '.tiff')
            output_path = self.directory.joinpath('circles', fish['ID'] + '.tiff')
            # Initialize Mask Generator
            masker = MaskGenerator(blur=int(fish['blur']),
                                   dp=int(fish['dp']),
                                   mindist=int(fish['mindist']),
                                   param1=int(fish['param1']),
                                   param2=int(fish['param2']))
            result = masker.run(bg_path, output_path)
            # Append to mask info
            mask_info['ID'].append(fish['ID'])
            mask_info['center_x'].append(result[0])
            mask_info['center_y'].append(result[1])
            mask_info['radius'].append(result[2])
        # Write center
        mask_info = pd.DataFrame(mask_info, columns=self.mask_columns)
        self.mask_data = self.mask_data.append(mask_info, ignore_index=True)
        self.mask_data.to_csv(self.mask_data_path, index=False)

    def set_thresholds(self, **kwargs):
        """Opens a GUI for setting tracking thresholds for each animal.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments used to initialize Tracker object.

        See Also
        --------
        behavior_analysis.tracking.tracker.Tracker
        """
        thresh1, thresh2 = 10, 200
        for i, fish in self.iter(missing_data='thresh1'):
            print(f'Set thresholds for fish {fish["ID"]}')
            bg_path = self.directory.joinpath('backgrounds', fish['ID'] + '.tiff')
            mask_path = self.directory.joinpath('circles', fish['ID'] + '.tiff')
            tracker = Tracker.from_background_path(bg_path, mask_path, thresh1, thresh2, **kwargs)
            (thresh1, thresh2) = SetThresholdsApp.start(tracker, fish['videos']['path'].values)
            self.animal_data.loc[i, 'thresh1'] = thresh1
            self.animal_data.loc[i, 'thresh2'] = thresh2
            self.animal_data.to_csv(self.animal_data_path, index=False)

    def track(self, parallel=True, **kwargs):
        """Track every untracked video for each animal.

        Parameters
        ----------
        parallel : bool (default = True)
            Whether or not to perform tracking for each video in parallel.

        Other Parameters
        ----------------
        n_processes : int
            Number of processes to run in parallel. Each video runs in a separate process.
        kwargs : dict
            Additional keyword arguments used to initialize Tracker object.

        See Also
        --------
        behavior_analysis.tracking.tracker.Tracker
        """
        # Tracking
        print_heading('TRACKING')
        for i, fish in self.iter():  # iterate through fish
            print_subheading(fish['ID'])
            try:
                assert not pd.isnull(fish['thresh1'])
            except AssertionError:
                print(f'Thresholds not set for {fish["ID"]}.')
                continue
            # Background path
            bg_path = self.directory.joinpath('backgrounds', fish['ID'] + '.tiff')
            circle_path = self.directory.joinpath('circles', fish['ID'] + '.tiff')
            # Initialize tracker object
            tracker = Tracker.from_background_path(bg_path, circle_path, fish['thresh1'], fish['thresh2'], **kwargs)
            # Generate inputs and outputs
            videos = fish['videos']
            untracked = videos[~videos['tracked']]
            video_paths = untracked['path'].values
            video_codes = untracked['code'].values
            outputs = [self.directory.joinpath('tracking', fish['ID'], code + '.csv') for code in video_codes]
            if not len(outputs):
                continue
            if parallel:  # run in parallel
                tracker.run_parallel(*list(zip(video_paths, outputs)), verbose=True)
            else:  # run each fish individually
                run_times = []
                for video_path, output_path in zip(video_paths, outputs):
                    print(output_path.stem + ':', end=' ')
                    _, t = tracker.run(video_path, output_path)
                    run_times.append(t)
                    print(Timer.convert_time(t))
                print(f'Mean time: {Timer.convert_time(np.mean(run_times))}', end='\n\n')
            self.animal_data.loc[i, 'tracking'] = True
            self.animal_data.to_csv(self.animal_data_path, index=False)
            # Check successfully tracked videos
            for video_code, tracking_output in zip(video_codes, outputs):
                if tracking_output.exists():
                    video_idx = self.video_data[self.video_data['code'] == video_code].iloc[0].name
                    self.video_data.loc[video_idx, 'tracked'] = True
            self.video_data.to_csv(self.video_data_path, index=False)

    def kinematics(self, parallel=True, **kwargs):
        """Extract egocentric kinematics from tracked data for each animal.

        Parameters
        ----------
        parallel : bool (default = True)
            Whether or not to extract kinematics from each tracking file in parallel.

        Other Parameters
        ----------------
        n_processes : int (default = 4)
            Number of processes to run in parallel. Each file runs in a separate process.

        See Also
        --------
        behavior_analysis.tracking.kinematics.Kinematics
        """
        # Kinematics
        print_heading('KINEMATICS')
        for i, fish in self.iter():  # iterate through fish
            print_subheading(fish['ID'])
            # Initialize kinematics object
            kinematics = Kinematics(kwargs.get('n_processes', 4))
            # Generate inputs and outputs
            tracking_directory = self.directory.joinpath('tracking', fish['ID'])
            output_directory = self.directory.joinpath('kinematics', fish['ID'])
            tracking_files = list(tracking_directory.glob('*.csv'))
            output_files = [output_directory.joinpath(f.name) for f in tracking_files]
            if parallel:  # run in parallel
                kinematics.run_parallel(*list(zip(tracking_files, output_files)), verbose=True)
            else:  # run each fish individually
                run_times = []
                for tracking_file, output_path in zip(tracking_files, output_files):
                    print(output_path.stem + ':', end=' ')
                    _, t = kinematics.run(tracking_file, output_path)
                    run_times.append(t)
                    print(Timer.convert_time(t))
                print(f'Mean time: {Timer.convert_time(np.mean(run_times))}', end='\n\n')
            self.animal_data.loc[i, 'kinematics'] = True
            self.animal_data.to_csv(self.animal_data_path, index=False)

    @property
    def kinematics_metadata(self):
        """Returns metadata for kinematic files."""
        kinematics_metadata = self.video_data.copy()
        kinematics_metadata = kinematics_metadata[kinematics_metadata['tracked']]
        for idx, video_info in kinematics_metadata.iterrows():
            path = self.directory.joinpath('kinematics', video_info.ID, video_info.code + '.csv')
            if not path.exists():
                path = np.nan
            kinematics_metadata.loc[idx, 'path'] = path
        kinematics_metadata = kinematics_metadata[~pd.isnull(kinematics_metadata['path'])]
        return kinematics_metadata

    def set_bout_detection_thresholds(self, **kwargs):
        """Set bout detection thresholds in the experiment metadata.

        Other Parameters
        ----------------
        threshold : float
            Default threshold for bout detection.
        winsize : float
            Default window size for bouts detection.

        Returns
        -------
        thresholds_changed : bool
            Whether or not bout detection thresholds have been changed in the experiment metadata.
        """
        # Get kinematics metadata
        metadata = self.kinematics_metadata
        if not len(metadata):
            raise ValueError('No kinematics files exist.')
        # Update kwargs from metadata
        if 'bout_detection' in self.metadata.keys():
            if 'threshold' not in kwargs.keys():
                kwargs['threshold'] = self.metadata['bout_detection']['threshold']
            if 'winsize' not in kwargs.keys():
                kwargs['winsize'] = self.metadata['bout_detection']['winsize']
        # Set thresholds
        ret, (threshold, winsize) = BoutDetectionThresholdApp.start(metadata, **kwargs)
        if ret == 1:
            thresholds_changed = False
            # Check whether new thresholds are different from thresholds stored in experiment metadata
            if 'bout_detection' in self.metadata.keys():
                if any([(self.metadata['bout_detection']['threshold'] != threshold),
                        (self.metadata['bout_detection']['winsize'] != winsize)]):
                    answer = self.yes_no_question('Continue with new bout detection thresholds?')
                    if answer:
                        thresholds_changed = True
                    else:
                        return False
            # Update experiment metadata with new thresholds
            self.metadata['bout_detection'] = dict(threshold=threshold, winsize=winsize)
            with open(self.metadata_path, 'w') as json_file:
                json.dump(self.metadata, json_file)
            return thresholds_changed
        else:  # thresholds not set
            sys.exit()

    def find_bouts(self, check_thresholds=True):
        """Segments bouts from kinematics files.

        Parameters
        ----------
        check_thresholds : bool (default = True)
            Open a GUI to set thresholds for bout segmentation.

        Returns
        -------
        df : pd.DataFrame
            Pandas DataFrame containing bout metadata.
        """
        if 'bout_detection' not in self.metadata.keys():
            check_thresholds = True
        if check_thresholds:
            thresholds_changed = self.set_bout_detection_thresholds()
        else:
            thresholds_changed = False
        # Get kinematics metadata
        metadata = self.kinematics_metadata
        if not len(metadata):
            raise ValueError('No kinematics files exist.')
        # Open pre-existing bouts or create new bouts file
        bout_columns = ['ID', 'code', 'start', 'end']
        bouts_path = self.directory.joinpath('analysis', 'bouts.csv')
        if not bouts_path.exists() or thresholds_changed:
            if not bouts_path.parent.exists():
                bouts_path.parent.mkdir()
            df = pd.DataFrame(columns=bout_columns)
        else:
            df = pd.read_csv(bouts_path, dtype=self.video_data_types)
        # Find bouts
        threshold = self.metadata['bout_detection']['threshold']
        winsize = self.metadata['bout_detection']['winsize']
        bout_detector = BoutDetector(threshold, winsize, 0)
        new_data = []
        for idx, info in metadata.iterrows():
            if ~np.isin(info.code, df['code'].values):
                print(info.code)
                kinematics = pd.read_csv(info.path)
                tip_angle = kinematics.loc[:, 'tip']
                bouts = bout_detector(tip_angle, frame_rate=info.fps)
                if len(bouts):
                    first_frames, last_frames = list(zip(*bouts))
                    IDs = [info.ID] * len(bouts)
                    codes = [info.code] * len(bouts)
                    video_bouts = pd.DataFrame(list(zip(IDs, codes, first_frames, last_frames)), columns=bout_columns)
                    new_data.append(video_bouts)
        df = pd.concat([df] + new_data, ignore_index=True)
        df.to_csv(bouts_path, index=False)
        return df

    def set_conditions(self, conditions=()):
        ret, out = SetConditionsApp.start(self.animal_data, conditions=conditions)
        if ret == 1:
            self.animal_data["condition"] = None
            for (ID, condition) in out.items():
                if condition is not None:
                    idx = self.animal_data[self.animal_data["ID"] == ID].squeeze().name
                    self.animal_data.loc[idx, "condition"] = condition
            self.animal_data.to_csv(self.animal_data_path, index=False)
            self.animal_columns = self.animal_columns + tuple(["condition"])
            print("Animal conditions saved.")
        else:
            print("Animal conditions not set.")

    def check_tracking(self, i):
        animal_data = self.get(i)
        for idx, video_info in animal_data['videos'].iterrows():
            tracking_path = self.subdirs['tracking'].joinpath(video_info.ID, video_info.code + '.csv')
            points_path = self.subdirs['tracking'].joinpath(video_info.ID, video_info.code + '.npy')
            video_path = video_info.path
            tracking = pd.read_csv(tracking_path)
            points = np.load(str(points_path))

            v = TrackedVideo(video_path, tracking=tracking, points=points)
            v.scroll()

            break
        #     print(animal_data.keys())
        # print(animal_data['videos']['path'])
