import logging
from pathlib import Path

import numpy as np
import pandas as pd
import vaex
from src.settings import DATA_DIRECTORY_AXA_CPT, TEXT_COLUMN

from prepare_data.utils import TextPreprocessor

preprocessor = TextPreprocessor(
    lower=True,
    remove_special_characters_mullenbach=True,
    remove_special_characters=False,
    remove_digits=True,
    remove_accents=False,
    remove_brackets=False,
    convert_danish_characters=False,
)

# The dataset requires a Licence in physionet. Once it is obtained, download the dataset with the following command in the terminal:
# wget -r -N -c -np --user <your_physionet_user_name> --ask-password https://physionet.org/files/mimiciii/1.4/
# Change the path of DOWNLOAD_DIRECTORY to the path where you downloaded mimiciii

logging.basicConfig(level=logging.INFO)


download_dir = Path(DATA_DIRECTORY_AXA_CPT)
output_dir = Path(DATA_DIRECTORY_AXA_CPT)

# Load the data
axa_df = pd.read_feather(download_dir / "axa_cpt.feather")

# Text preprocess the notes
with vaex.cache.memory_infinite():  # pylint: disable=not-context-manager
    axa_df = vaex.from_pandas(axa_df)
    axa_df = preprocessor(axa_df)
    axa_df = axa_df.to_pandas_df()

# save files to disk
axa_df.to_feather(output_dir / "axa_cpt.feather")
