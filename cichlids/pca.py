from behavior_analysis.experiment import BehaviorExperiment
from behavior_analysis.analysis.pca import PCA
from behavior_analysis.analysis.csv import import_csvs
import numpy as np


if __name__ == "__main__":
    # Open experiment
    experiment = BehaviorExperiment.open(r"D:\DATA\cichlids\a_burtoni")
    output_dir = experiment.directory.joinpath('analysis')
    if not output_dir.exists():
        output_dir.mkdir()
    # Import data
    df = import_csvs(experiment.directory.joinpath('kinematics'))
    df = df[df['tracked']]
    k_cols = [col for col in df.columns if col.startswith('k')]
    df = df[k_cols]
    # Run pca
    pca = PCA(df)
    mean, std = pca.mean, pca.std
    tail_stats = np.array([mean, std])
    np.save(output_dir.joinpath("tail_statistics.npy"), tail_stats)
    transformed, pca_obj = pca.transform(whiten=True, mean=mean, std=std)
    explained_var = pca_obj.explained_variance_ratio_
    eigenfish = pca_obj.components_
    np.save(output_dir.joinpath("explained_variance.npy"), explained_var)
    np.save(output_dir.joinpath("eigenfish.npy"), eigenfish)
