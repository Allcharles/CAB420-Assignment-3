from os import listdir, remove
from os.path import isfile, join
from math import inf
import numpy as np
import re

BATCH_SIZE = 1000
FOLDER_PATH = "D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/train/"

# Regex patterns to determine the instrument family and its type (acoustic/electronic)
instrument_family_pattern = r'(\w+)_\w+_\d+-\d+-\d+.png$'
instrument_type_pattern = r'\w+_(\w+)_\d+-\d+-\d+.png$'

# Delete previous filelist
try:
    remove(FOLDER_PATH + "filelist.csv")
except:
    print("No Previous filelist.csv file")

# Grab list of files inside folder
files = [f for f in listdir(FOLDER_PATH) if isfile(join(FOLDER_PATH, f))]

# Stop train size from being too large
if (BATCH_SIZE > len(files)):
    BATCH_SIZE = len(files)

# Grab list of files to train on
output = []
for file_index in np.arange(0, len(files), len(files)/BATCH_SIZE):
    instrument_family = re.match(
        instrument_family_pattern, files[int(file_index)])
    instrument_type = re.match(instrument_type_pattern, files[int(file_index)])
    output.append("{},{},{}\n".format(
        files[int(file_index)], instrument_family.group(1), instrument_type.group(1)))

# Create filelist
with open(FOLDER_PATH + "filelist.csv", "w+") as f:
    for details in output:
        f.write(details)
