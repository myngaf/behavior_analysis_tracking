from .csv import import_csvs
from .pca import PCA
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Union


class BoutData(PCA):
    """Data object for handling bout kinematics.

    Attributes
    ----------
    data : pd.DataFrame
        Multi-indexed DataFrame containing bout data.
    metadata : pd.DataFrame
        DataFrame containing basic information about all bouts.
    """

    def __init__(self, data: pd.DataFrame, metadata: pd.DataFrame = None):
        super().__init__(data)
        assert isinstance(self.data.index, pd.MultiIndex), 'Data must be MultiIndexed!'
        self.metadata = metadata

    @classmethod
    def from_metadata(cls,
                      metadata: Union[pd.DataFrame, str, Path],
                      directory: Union[str, Path],
                      tail_only: bool = True):
        """Constructor method for generating BoutData from metadata.

        Parameters
        ----------
        metadata: pd.DataFrame | str | Path
            Either a DataFrame object containing bout metadata or a path to a corresponding csv file.
        directory: str | Path
            Top-level directory containing kinematic data.
        tail_only : bool (default = True)
            Whether or not to only keep information about tail kinematics when importing bouts.
        """
        # Open the metadata DataFrame
        if isinstance(metadata, pd.DataFrame):
            bouts_df = metadata
        elif isinstance(metadata, (str, Path)):
            bouts_df = pd.read_csv(metadata, dtype=dict(ID=str, code=str))
        else:
            raise TypeError('metadata must be path or DataFrame')
        # Get paths
        directory = Path(directory)
        paths = []
        for ID, fish_bouts in bouts_df.groupby('ID'):
            for code in pd.unique(fish_bouts['code']):
                paths.append(directory.joinpath(ID, code + '.csv'))
        # Import bouts
        data = import_csvs(*paths)
        # Keep desired columns
        print('Reformatting multi-indexed DataFrame...', end=' ')
        if 'tracked' in data.columns:
            data = data[data['tracked']]
        if tail_only:
            tail_columns = [col for col in data.columns if col[0] == 'k']
            data = data[tail_columns]
        # Assign bout index
        index_df = data.index.to_frame()
        video_data_dfs = []
        for code, video_bouts in bouts_df.groupby('code'):
            video_data = index_df.loc[(slice(None), code, slice(None)), :].copy()
            bout_index = np.empty(len(video_data)) + np.nan
            for idx, info in video_bouts.iterrows():
                bout_index[info.start:info.end + 1] = idx
            video_data['bout_index'] = bout_index
            video_data_dfs.append(video_data)
        concat = pd.concat(video_data_dfs)
        reindexed = pd.MultiIndex.from_frame(concat)
        data.index = reindexed.reorder_levels(('ID', 'code', 'bout_index', 'frame'))
        data = data[~data.index.get_level_values('bout_index').isna()]
        data.index.set_levels(data.index.levels[2].astype('int64'), level='bout_index', inplace=True)
        print('done!\n')
        # Return object
        return cls(data, bouts_df)

    def __str__(self):
        return self.data.__str__()

    def filter_tail_lengths(self, percentile=99):
        print('Filtering tail lengths...', end=' ')
        if 'length' not in self.data.columns:
            raise ValueError('Data des not contain "length" column.')
        long_tail = self.data[self.data['length'] > np.percentile(self.data['length'].values, percentile)]
        long_tail_bouts = long_tail.index.get_level_values('bout_index')
        metadata = self.metadata.loc[self.metadata.index[np.isin(self.metadata.index, long_tail_bouts, invert=True)]]
        data = self.data[np.isin(self.data.index.get_level_values('bout_index'), long_tail_bouts, invert=True)]
        print(f'{len(long_tail_bouts.unique())} bouts removed.\n')
        return BoutData(data, metadata)

    def tail_only(self):
        tail_columns = [col for col in self.data.columns if col[0] == 'k']
        data = self.data[tail_columns]
        return BoutData(data, self.metadata)

    def _get_from_frame(self, df):
        indexer = self.data.index.get_locs([df['ID'].values.unique(),
                                            df['code'].values.unique(),
                                            df.index])
        sliced = self.data.iloc[indexer, :]
        return BoutData(sliced, df)

    def _get_from_dict(self, idxs=(), **kwargs):
        index_values = dict(IDs=slice(None), codes=slice(None), bout_indices=slice(None))
        for key, val in kwargs.items():
            if len(val):
                index_values[key] = val
        indexer = self.data.index.get_locs([index_values['IDs'], index_values['codes'], index_values['bout_indices']])
        if len(idxs):
            indices = self.data.index.take(indexer).remove_unused_levels()
            take_bouts = indices.levels[2].take(idxs)
            take_indices = indices.get_locs([slice(None), slice(None), take_bouts])
            indexer = indexer[take_indices]
        sliced = self.data.iloc[indexer, :]
        sliced.index = sliced.index.remove_unused_levels()
        if self.metadata is not None:
            metadata = self.metadata.loc[sliced.index.levels[2]]
        else:
            metadata = None
        return BoutData(sliced, metadata)

    def get(self, IDs=(), codes=(), bout_indices=(), idxs=(), df=None):
        if df is not None:
            new = self._get_from_frame(df)
        else:
            new = self._get_from_dict(IDs=IDs, codes=codes, bout_indices=bout_indices, idxs=idxs)
        return new

    def iter(self, values=False, ndims=None, **kwargs):
        data = self.data
        if len(kwargs):
            data = self.get(**kwargs).data
        for idx, bout in data.groupby('bout_index'):
            if values:
                bout = bout.values
                if ndims:
                    bout = bout[:, :ndims]
                yield idx, bout
            else:
                yield idx, bout.reset_index(level=['ID', 'code', 'bout_index'], drop=True)

    def to_list(self, values=False, ndims=None, **kwargs):
        return [bout for i, bout in self.iter(values, ndims, **kwargs)]

    def whiten(self, **kwargs):
        whitened = super().whiten(**kwargs).data
        return BoutData(whitened, metadata=self.metadata)

    def transform(self, **kwargs):
        transformed, pca = super().transform(**kwargs)
        transformed = pd.DataFrame(transformed.data, index=self.data.index)
        return BoutData(transformed, metadata=self.metadata), pca

    def map(self, vectors, **kwargs):
        mapped = super().map(vectors, **kwargs).data
        mapped = pd.DataFrame(mapped, index=self.data.index, columns=[f'c{i}' for i in range(mapped.shape[1])])
        return BoutData(mapped, metadata=self.metadata)
