import os
import tkinter
from tkinter import filedialog, messagebox
import sys


def create_folder(directory, name):
    """Creates a new folder

    Parameters
    ----------
    directory : str
        Directory where folder should be created
    name : str
        Name of the new folder

    Returns
    -------
    new_path : str
        Path to the new folder
    """
    new_path = os.path.join(directory, name)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    return new_path


def create_filepath(directory, basename, suffix, return_exists=False):
    """Creates a file path

    Parameters
    ----------
    directory : str
        Directory where path should be created
    basename : str
        Name of the new file
    suffix : str
        Suffix that is appended to the name of the file (e.g. '.tiff')
    return_exists : bool, optional (default = False)
        Return a boolean as a second argument stating if the file already exists

    Returns
    -------
    path_name [, path_exists] : str [, bool]
        Path to the new file, if return_exists==True returns whether path already exists as second argument
    """
    file_name = ''.join([basename, suffix])
    path_name = os.path.join(directory, file_name)
    if return_exists:
        exists = os.path.exists(path_name)
        return path_name, exists
    else:
        return path_name


def get_files(directory, return_paths=True):
    """Lists the files in a directory sorted alphabetically

    N.B. For getting folders in a directory use get_directories

    Parameters
    ----------
    directory : str
        Path to a directory containing files
    return_paths : bool, optional (default = True)
        Return paths as well as file names

    Returns
    -------
    file_names [, file_paths] : list [, list]
        Files in a directory (list of strings), if return_paths==True returns complete paths as second argument
    """
    filenames = os.listdir(directory)
    remove_files = ['.DS_Store', 'Thumbs.db']
    for filename in remove_files:
        if filename in filenames:
            filenames.remove(filename)
    filenames = [filename for filename in filenames if os.path.isfile(os.path.join(directory, filename))]
    filenames.sort()
    if return_paths:
        filepaths = [os.path.join(directory, name) for name in filenames]
        return filenames, filepaths
    else:
        return filenames


def get_directories(directory, return_paths=True):
    """Lists the folders in a directory sorted alphabetically

    N.B. For getting files in a directory use get_files

    Parameters
    ----------
    directory : str
        Path to a directory containing folders
    return_paths : bool, optional (default = True)
        Return paths as well as folder names

    Returns
    -------
    dirnames [, file_paths] : list [, list]
        Folders in a directory (list of strings), if return_paths==True returns complete paths as second argument
    """
    dirnames = os.listdir(directory)
    dirnames = [dirname for dirname in dirnames if os.path.isdir(os.path.join(directory, dirname))]
    dirnames.sort()
    if return_paths:
        dirpaths = [os.path.join(directory, name) for name in dirnames]
        return dirnames, dirpaths
    else:
        return dirnames


# ====================
# FILEPICKER FUNCTIONS
# ====================


def pickfile(path='.'):  # add filetypes
    root = tkinter.Tk()
    root.withdraw()
    f = filedialog.askopenfilename(parent=root, title='Choose a file')
    if f:
        root.destroy()
        del root
        return f
    else:
        print("No file picked, exiting!")
        root.destroy()
        del root
        sys.exit()


def saveasfile(path='.', filetypes=[], defaultextension=''):  # add filetypes
    root = tkinter.Tk()
    root.withdraw()
    f = filedialog.asksaveasfilename(parent=root, title='Choose a filepath to save as', filetypes=filetypes,
                                       defaultextension=defaultextension)
    if f:
        root.destroy()
        del root
        return f
    else:
        print("No file picked, exiting!")
        root.destroy()
        del root
        sys.exit()


def pickfiles(path='.', filetypes=[], defaultextension=''):
    root = tkinter.Tk()
    root.withdraw()
    f = filedialog.askopenfilenames(parent=root, title='Choose a file', filetypes=filetypes)
    if f:
        f = root.tk.splitlist(f)
        root.destroy()
        del root
        return f
    else:
        print("No file picked, exiting!")
        root.destroy()
        del root
        sys.exit()


def pickdir(path='.'):
    root = tkinter.Tk()
    root.withdraw()
    dirname = filedialog.askdirectory(parent=root, initialdir=".", title='Please select a directory')

    root.destroy()
    if len(dirname) > 0:
        return dirname
    else:
        print("No directory picked, exiting!")
        sys.exit()


def askyesno(title='Display?', text="Use interactive plotting?"):
    root = tkinter.Tk()
    root.withdraw()
    tf = messagebox.askyesno(title, text)
    root.destroy()
    return tf
