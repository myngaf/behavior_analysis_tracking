import cv2
import pandas as pd


def print_heading(s):
    print('\n' + '=' * len(s))
    print(s)
    print('=' * len(s), '\n')


def print_subheading(s):
    print('\n' + s)
    print('-' * len(s))


def yes_no_question(q):
    """Asks the user a yes/no question

    Valid affirmative answers (case insensitive): 'y', 'yes', '1', 't', 'true'
    Valid negative answers (case insensitive): 'n', 'no', '0', 'f', 'false'

    Parameters
    ----------
    q : str
        A question that the user can answer in the command line

    Returns
    -------
    bool : True for affirmative answers, False for negative answers

    Raises
    ------
    ValueError
        If unrecognised answer given
    """
    affirmative_answers = ['y', 'yes', '1', 't', 'true']
    negative_answers = ['n', 'no', '0', 'f', 'false']
    answer = input(q + ' [y/n] ')
    if answer.lower() in affirmative_answers:
        return True
    elif answer.lower() in negative_answers:
        return False
    else:
        raise ValueError('Invalid answer! Recognised responses: {}'.format(affirmative_answers + negative_answers))


class KeyboardInteraction(object):
    """Class for handling keyboard interaction

    Class Attributes
    ----------------
    enter : 13
        ASCII code for enter/carriage return key
    esc : 27
        ASCII code for escape key
    space : 32
        ASCII code for escape key"""

    enter_key = 13
    esc_key = 27
    space_key = 32

    def __init__(self):
        self.k = None

    def wait(self, t=0):
        self.k = cv2.waitKey(t)

    def enter(self):
        return self.k == self.enter_key

    def space(self):
        return self.k == self.space_key

    def esc(self):
        return self.k == self.esc_key

    def valid(self):
        """Checks whether the enter, escape or space keys were pressed

        Parameters
        ----------
        k : int
            ASCII key code

        Returns
        -------
        bool
        """
        if self.k in (self.enter_key, self.esc_key, self.space_key):
            return True
        else:
            return False


def read_csv(path, **kwargs):
    """Opens a csv file as a DataFrame

    Parameters
    ----------
    path : str
        The path to a csv file.

    kwargs : dict, optional
        (Column, callable) pairs for reading the csv file.

    Returns
    -------
    df : pd.DataFrame
        A correctly formatted DataFrame.
    """
    df = pd.read_csv(path)
    for col, read_as in kwargs.items():
        existing_data = df[col][~df[col].isnull()]
        df.loc[existing_data.index, col] = existing_data.apply(read_as)
    return df


class PrintProgress:

    def __init__(self, n):
        self.n = n
        self.deci = int(self.n / 10)
        self.use_deci = (self.n >= 10)
        self.centi = int(self.n / 100)
        self.use_centi = (self.n >= 100)

    def __call__(self, i):
        if i > 0:
            if self.use_centi:
                if (i % self.deci) == 0:
                    print(f'# {(i // self.deci) * 10}%')
                elif (i % self.centi) == 0:
                    print('#', end='')
            elif self.use_deci and ((i % self.deci) == 0):
                print('#', end='')
            elif i == (self.n - 1):
                print('########## 100%')
