from typing import List

import numpy as np
import uproot as ur
from tqdm import tqdm


def preprocess_input(analysis_names: List[str],
                     file_paths: List[str]):
    # list for storing events of corresponding SR
    event_SR_matrix_list: List[np.array] = []
    # names of the signal regions
    SR_names: List[str] = []
    print("Files preprocessing:\n")
    for _, file_path in enumerate(tqdm(file_paths)):
        # opening the file
        with ur.open(file_path) as file:
            # access to the ntuple structure
            ttree = file["ntuple"]
            # signal regions are the keys of the ttree
            signal_regions = ttree.keys()
            # signal regions with counts of events
            # that passed
            ttree_arrays = ttree.arrays()
            # empty matrix to store data in numpy style
            events: np.array = np.empty(
                shape=(len(ttree_arrays), len(signal_regions)),
                dtype=np.float32)
            # iterating through signal regions to extract the
            # arrays and store row wise
            for sr_idx, sr in enumerate(signal_regions):
                SR_names.append(sr)
                events[:, sr_idx] = np.array(
                    ttree_arrays[sr], dtype=np.float32)
            # list of matrices
            event_SR_matrix_list.append(events)

    # concatenate the matrices
    event_SR_matrix_combined: np.array = np.concatenate(
        event_SR_matrix_list,
        axis=1,
        dtype=np.float32)
    print(f"\nNumber of events: {event_SR_matrix_combined.shape[0]}\n"
          "Number of SRs: "
          f"{event_SR_matrix_combined.shape[1] - len(analysis_names)*2}.")

    return event_SR_matrix_combined, SR_names
