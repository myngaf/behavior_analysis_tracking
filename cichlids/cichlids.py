from behavior_analysis.experiment import BehaviorExperiment
# from behavior_analysis.analysis.bouts import BoutData
# from behavior_analysis.analysis.bout_mapping import calculate_distance_matrix
# import pandas as pd
# from matplotlib import pyplot as plt
# from scipy.spatial.distance import squareform
from pathlib import Path
import numpy as np
# from behavior_analysis.analysis.bout_mapping import calculate_distance_matrix_templates, interpolate_nd
# from matplotlib import pyplot as plt


if __name__ == "__main__":

    # Open experiment
    experiment = BehaviorExperiment.open(r"D:\DATA\cichlids\a_burtoni",
                                         video_directory=r"J:\_Projects\Cichlid_Group\Duncan\prey_capture_experiments\a_burtoni\videos")
    print(experiment)
    # # Update animal info
    # experiment.update()
    # # Calculate backgrounds
    # experiment.calculate_backgrounds(parallel=True, n_processes=4)
    # # Set tracking thresholds
    # experiment.set_thresholds()
    # # Run tracking
    # experiment.track(parallel=True, n_processes=4)
    # # Run kinematics
    # experiment.kinematics(parallel=True, n_processes=4)
    experiment.find_bouts(check_thresholds=True)
