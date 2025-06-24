from behavior_analysis.experiment import BehaviorExperiment
from pathlib import Path
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

if __name__ == "__main__":

    path = Path(r"C:\Users\manyung.ng\Documents\behaviour_analysis\behavior_analysis_tracking\recordings")

    # Create experiment
    # experiment = BehaviorExperiment.create(path) # only do this once

    # Open experiment
    experiment = BehaviorExperiment.open(path)
    print(experiment)
    #
    # # Update animal info
    # experiment.update()
    #
    # # Calculate background
    # experiment.calculate_backgrounds(parallel=True, n_processes=20)

    # Set tracking threshold
    experiment.set_thresholds()

    # Run tracking
    experiment.track(parallel=True, n_processes=20)
    #
    # Run kinematics
    experiment.kinematics(parallel=True, n_processes=20)

    # Extract bouts
    experiment.find_bouts(check_thresholds=False)