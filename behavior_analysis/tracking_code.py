from behavior_analysis.experiment import BehaviorExperiment
from pathlib import Path
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def run_tracking(paths):
    for path in paths:
        # Create experiment
        # experiment = BehaviorExperiment.create(path) # only do this once

        # Open experiment
        experiment = BehaviorExperiment.open(path)
        print(experiment)
        #
        # Update animal info
        # experiment.update()
        #
        # # Calculate background
        # experiment.calculate_backgrounds(parallel=True, n_processes=20)
        #
        # # Set circular mask threshold
        experiment.set_circular_mask()
        #
        # # Calculate mask
        experiment.calculate_circular_mask()
        #
        # Set tracking threshold
        experiment.set_thresholds()
        #
        # Run tracking
        experiment.track(parallel=True, n_processes=20)
        #
        # Run kinematics
        experiment.kinematics(parallel=True, n_processes=20)
        #
        # Extract bouts
        experiment.find_bouts(check_thresholds=False)

if __name__ == "__main__":

    path = [Path(r"E:\Joyce\test_experiment")]
    run_tracking(path)



