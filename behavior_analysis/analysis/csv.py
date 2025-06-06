import pandas as pd
from pathlib import Path
from ..utilities.io import PrintProgress


def import_csvs(*args, index_levels=(1, 0), level_names=("ID", "code", "frame")):
    """Imports and concatenates csv files in a multi-indexed DataFrame.

    Parameters
    ----------
    args : iterable of Path or string objects
        Files or directories containing csv files to be concatenated.
    index_levels : iterable
        Iterable of ints. Each index level represents the part of the file path to be included in the multi-index. E.g.
        index level 0 indicates the filename (excluding the extension), index level 1 indicates the parent folder etc..
    level_names : iterable
        Iterable of strings. Names for each index level + the name of the index within each csv.

    Returns
    -------
    pd.DataFrame
        A multi-indexed DataFrame. Each level of the index represents a part of the file path (given by the index level)
        of the csv files included in the DataFrame."""
    try:  # check index levels and level names match
        assert len(level_names) == (len(index_levels) + 1)
    except AssertionError:
        raise ValueError("Number of level names must be one greater than number of index levels.")
    # Obtain file paths
    files = []
    for arg in args:
        path = Path(arg)
        try:  # check path exists
            assert path.exists()
        except AssertionError:
            print(f'Path: {path} does not exist. Skipping.')
            continue
        if path.is_dir():  # if path is a directory, search this directory and all sub-directories for csv files
            for f in path.glob('**/*.csv'):
                files.append(f)
        elif path.suffix == '.csv':  # if path is a csv file, add it to the list of files
            files.append(path)
        else:
            print(f'Path: {path} is not a valid csv file. Skipping.')
    # Open all valid csv files
    print(f'Opening {len(files)} csv files...')
    dfs = []
    indexer = []
    progress = PrintProgress(len(files))
    for i, f in enumerate(files):
        parts = [f.stem] + list(reversed(f.parent.parts))  # obtain parts of the path
        df = pd.read_csv(f)  # open the DataFrame
        dfs.append(df)  # append DataFrame to list of DataFrames
        indexer.append(tuple([parts[i] for i in index_levels]))  # add parts of the file path specified by the index
        # levels to the indexer
        progress(i)
    concatenated = pd.concat(dfs, keys=indexer, names=level_names)  # concatenate DataFrames
    print('done!')
    print(f'DataFrame has shape: {concatenated.shape}.\n'
          f'Index levels: {level_names}.\n')
    return concatenated
