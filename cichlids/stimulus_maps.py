from behavior_analysis.experiment import BehaviorExperiment
import pandas as pd


if __name__ == "__main__":
    # Open experiment
    experiment = BehaviorExperiment.open(r"D:\DATA\cichlids\l_attenuatus",
                                         video_directory=r"J:\_Projects\Cichlid_Group\Duncan\prey_capture_experiments\l_attenuatus\videos")
    # Import kinematic data
    hunting_events = pd.read_csv(experiment.subdirs["analysis"].joinpath("hunting_events.csv"),
                                 dtype={"ID": str, "code": str})

    for event in hunting_events:
        tracking_path = experiment.subdirs["tr"]
