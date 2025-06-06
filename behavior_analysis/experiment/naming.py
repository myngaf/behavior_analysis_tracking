from pathlib import Path


def generate_ID(YYYY_MM_DD, idx):
    """Creates a unique ID from the date and fish name.

    Parameters
    ----------
    YYYY_MM_DD : str
        Date in the format YYYY_MM_DD (e.g. '2016_01_30')
    idx : str or int
        Fish name (e.g. 'fish3', 'fish12') or index of the fish in the experiment

    Returns
    -------
    ID : str
        Fish ID in the format 'YYYYMMDDNN' (year, month, day, zero-padded fish number)

    Raises
    ------
    TypeError if idx anything other than a string or int.
    """
    date_ID = YYYY_MM_DD.replace('_', '')
    if type(idx) == int:
        NN = str(idx)
    elif type(idx) == str:
        NN = ''.join(filter(str.isdigit, idx))
    else:
        raise TypeError('idx must be integer or string')
    if len(NN) == 1:
        NN = '0' + NN
    ID = date_ID + NN
    return ID


def generate_video_code(fish_ID, video_path):
    """Creates a unique ID from the fish ID and a video file

    Parameters
    ----------
    fish_ID : str
        Unique fish identifier
    video_path : str
        A timestamped video file in the format hour-minute-second separated by '-' (e.g.'12-34-56.avi')

    Returns
    -------
    video_code : str
        Video code in the format 'IDHHMMSS' (fish_ID, hour, minute, second)
    """
    timestamp = Path(video_path).stem
    timestamp = timestamp.replace('-', '')
    video_code = fish_ID + timestamp
    return video_code
