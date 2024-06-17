import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple

import h5py
import numpy as np
import scipy
import tifffile
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from caload import base, utils

log = logging.getLogger(__name__)


def _recording_id_from_path(path: Union[Path, str]) -> Tuple[str, str]:
    return Path(path).as_posix().split('/')[-1].split('_') # type: ignore


def _animal_id_from_path(path: Union[Path, str]) -> str:
    return Path(path).as_posix().split('/')[-2]


def create_analysis(analysis_path: str, data_root: str) -> base.Analysis:

    # Create analysis data folder
    if not os.path.exists(analysis_path):
        os.mkdir(analysis_path)

    # Create analysis
    print(f'Create analysis at {analysis_path}')
    analysis = base.Analysis(analysis_path, mode=base.Mode.create)

    # Scan for data folders
    recording_folders = scan_folder(data_root, [])

    # Start digesting recordings
    digest_folder(recording_folders, analysis)


def scan_folder(root_path: str, recording_list: List[str]) -> List[str]:

    for fld in os.listdir(root_path):
        current_path = os.path.join(root_path, fld)

        # Skip files
        if not os.path.isdir(current_path):
            continue

        # Skip analysis directory
        if os.path.exists(os.path.join(current_path, 'metadata.db')):
            continue

        if 'suite2p' in os.listdir(current_path):
            recording_list.append(current_path)
            continue

        recording_list = scan_folder(current_path, recording_list)

    return recording_list


def digest_folder(folder_list: List[str], analysis: base.Analysis):

    print(f'Process folders: {folder_list}')
    for current_path in folder_list:
        print(f'Recording folder {current_path}')

        # Get animal
        animal_id = _animal_id_from_path(current_path)
        _animal_list = analysis.animal(animal_id=animal_id)
        # Add new animal
        if len(_animal_list) == 0:
            animal = analysis.add_animal(animal_id=animal_id)
            animal['animal_id'] = animal_id
        else:
            animal = _animal_list[0]

        # Create debug folder
        debug_folder_path = os.path.join(current_path, 'debug')
        if not os.path.exists(debug_folder_path):
            os.mkdir(debug_folder_path)

        # Get recording
        # Expected recording folder format "<rec_date('YYYY-mm-dd'>_<rec_id>"
        rec_date, rec_id, *_ = _recording_id_from_path(current_path)
        _recording_list = analysis.recordings(animal_id=animal.id, rec_date=rec_date, rec_id=rec_id)
        # Add recording
        if len(_recording_list) > 0:
            print('Recording already exists. Skip')
            continue

        recording = animal.add_recording(rec_date=rec_date, rec_id=rec_id)
        recording['rec_date'] = utils.parse_date(rec_date)
        recording['rec_id'] = rec_id

        # Load s2p processed data
        s2p_path = os.path.join(current_path, 'suite2p', 'plane0')

        # Load suite2p's analysis options
        print('Include suite2p ops')
        ops = np.load(os.path.join(s2p_path, 'ops.npy'), allow_pickle=True).item()
        unravel_dict(ops, recording)

        print('Calculate frame timing of signal')
        with h5py.File(os.path.join(current_path, 'Io.hdf5'), 'r') as io_file:

            mirror_position = np.squeeze(io_file['ai_y_mirror_in'])[:]
            mirror_time = np.squeeze(io_file['ai_y_mirror_in_time'])[:]

            # Calculate frame timing
            frame_idcs, frame_times = calculate_ca_frame_times(mirror_position, mirror_time)

            record_group_ids = io_file['__record_group_id'][:].squeeze()
            record_group_ids_time = io_file['__time'][:].squeeze()

            ca_rec_group_id_fun = scipy.interpolate.interp1d(record_group_ids_time, record_group_ids, kind='nearest')

        # Plot y mirror signal with detected frames
        plot_y_mirror_debug_info(mirror_position, mirror_time, frame_idcs, current_path)

        print('Load ROI data')
        fluorescence = np.load(os.path.join(s2p_path, 'F.npy'), allow_pickle=True)
        spikes_all = np.load(os.path.join(s2p_path, 'spks.npy'), allow_pickle=True)
        roi_stats_all = np.load(os.path.join(s2p_path, 'stat.npy'), allow_pickle=True)
        iscell_all = np.load(os.path.join(s2p_path, 'iscell.npy'), allow_pickle=True)

        # Check if frame times and signal match
        if frame_times.shape[0] != fluorescence.shape[1]:
            print(f'Detected frame times\' length doesn\'t match frame count. '
                        f'Detected frame times ({frame_times.shape[0]}) / Frames ({fluorescence.shape[1]})')

            # Shorten signal
            if frame_times.shape[0] < fluorescence.shape[1]:
                fluorescence = fluorescence[:, :frame_times.shape[0]]
                print('Truncated signal at end to resolve mismatch. Check debug output to verify')

            # Shorten frame times
            else:
                frame_times = frame_times[:fluorescence.shape[1]]
                print('Truncated detected frame times at end to resolve mismatch. Check debug output to verify')

        # Get imaging rate from sync signal
        imaging_rate = 1./np.mean(np.diff(frame_times))  # Hz
        print(f'Estimated imaging rate {imaging_rate:.2f}Hz')

        # Save to recording
        recording['roi_num'] = fluorescence.shape[0]
        recording['signal_length'] = fluorescence.shape[0]
        recording['imaging_rate'] = imaging_rate
        recording['ca_times'] = frame_times
        recording['record_group_ids'] = ca_rec_group_id_fun(frame_times)

        # Add suite2p's analysis ROI stats
        print('Add ROI stats and calculate signals')
        for roi_id in tqdm(range(fluorescence.shape[0])):
            # Create ROI
            roi = recording.add_roi(roi_id=roi_id)

            roi_stats = roi_stats_all[roi_id]

            # Write ROI stats
            roi.update({f's2p/{k}': v for k, v in roi_stats.items()})

            # Write data
            fluores = fluorescence[roi_id]
            spikes = spikes_all[roi_id]
            iscell = iscell_all[roi_id]
            roi['fluorescence'] = fluores
            roi['spikes'] = spikes
            roi['iscell'] = iscell

        print('Add display phase data')
        with h5py.File(os.path.join(current_path, 'Display.hdf5'), 'r') as disp_file:

            # Get attributes
            recording.update({f'display_data/attrs/{k}': v for k, v in disp_file.attrs.items()})

            for key1, member1 in tqdm(disp_file.items()):

                # If dataset, write to file
                if isinstance(member1, h5py.Dataset):
                    recording[f'display_data/{key1}'] = member1[:]
                    continue

                # Otherwise it's a group -> keep going

                # Add phase
                if 'phase' in key1:

                    phase = recording.add_phase(phase_id=int(key1.replace('phase', '')))
                    # Write attributes
                    phase.update({k: v for k, v in member1.attrs.items()})

                    # Write datasets
                    for key2, member2 in member1.items():
                        if isinstance(member2, h5py.Dataset):
                            phase[key2] = member2[:]

                # Add other data
                else:
                    # Write attributes
                    recording.update({f'display/{key1}/{k}': v for k, v in member1.attrs.items()})

                    # Get datasets
                    for key2, member2 in member1.items():
                        if isinstance(member2, h5py.Dataset):
                            recording[f'display/{key1}/{key2}'] = member2[:]

            analysis.session.commit()


def load_metadata(folder_path: str) -> Dict[str, Any]:
    """Function searches for and returns metadata on a given folder path

    Function scans the `folder_path` for metadata yaml files (ending in `meta.yaml`)
    and returns a dictionary containing their contents
    """

    meta_files = [f for f in os.listdir(folder_path) if f.endswith('meta.yaml')]

    log.info(f'Found {len(meta_files)} metadata files in {folder_path}.')

    meta_data = {}
    for f in meta_files:
        with open(os.path.join(folder_path, f), 'r') as stream:
            try:
                meta_data.update(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)

    return meta_data


def unravel_dict(dict_data: dict, entity: Union[base.Animal, base.Recording, base.Roi]):
    for key, item in dict_data.items():
        if isinstance(item, dict):
            unravel_dict(item, entity)
            continue
        entity[key] = item


def calculate_ca_frame_times(mirror_position: np.ndarray, mirror_time: np.ndarray):

    peak_prominence = (mirror_position.max() - mirror_position.min()) / 4
    peak_idcs, _ = scipy.signal.find_peaks(mirror_position, prominence=peak_prominence)
    trough_idcs, _ = scipy.signal.find_peaks(-mirror_position, prominence=peak_prominence)

    # Find first trough
    first_peak = peak_idcs[0]
    first_trough = trough_idcs[trough_idcs < first_peak][-1]

    # Discard all before (and including) first trough
    trough_idcs = trough_idcs[first_trough < trough_idcs]
    frame_idcs = np.sort(np.concatenate([trough_idcs, peak_idcs]))

    # Get corresponding times
    frame_times = mirror_time[frame_idcs]

    return frame_idcs, frame_times


def plot_y_mirror_debug_info(mirror_position: np.ndarray, mirror_time: np.ndarray,
                             frame_idcs: np.ndarray, recording_path: str):

    # Plot frame time detection results
    fig_name = 'frame_timing_detection'
    fig, ax = plt.subplots(1, 3, figsize=(18, 4), num=fig_name)

    frame_num = 30

    markersize = 3.
    start_times = mirror_time < mirror_time[frame_idcs[frame_num]]
    ax[0].plot(mirror_time[start_times], mirror_position[start_times], color='blue')
    ax[0].plot(mirror_time[frame_idcs[:frame_num]], mirror_position[frame_idcs[:frame_num]], 'o', color='red', markersize=markersize)
    ax[0].set_xlim(mirror_time[0], mirror_time[frame_idcs[frame_num]])

    ax[1].hist(np.diff(mirror_time[frame_idcs]))

    end_times = mirror_time > mirror_time[frame_idcs[-frame_num]]
    ax[2].plot(mirror_time[end_times], mirror_position[end_times], color='blue')
    ax[2].plot(mirror_time[frame_idcs[-frame_num:]], mirror_position[frame_idcs[-frame_num:]], 'o', color='red', markersize=markersize)
    ax[2].set_xlim(mirror_time[frame_idcs[-frame_num]], mirror_time[-1])

    fig.tight_layout()
    plt.savefig(os.path.join(recording_path, 'debug', f'{fig_name}.pdf'), format='pdf')
    plt.clf()


if __name__ == '__main__':
    # base_path = '../../hladnik_arrenberg/hladnik_arrenberg_code/moving_rf_mapping/data/default_orientation'
    # base_path = '../../../../TEMP_testdata/default_orientation'
    # base_path = '//172.25.250.112/thladnik/hladnik_arrenberg/data/localized_motion_rfs'
    # base_path = 'Z:/hladnik_arrenberg/data/localized_motion_rfs'

    data_root = 'D:/data/hladnik_arrenberg/localized_motion_rfs'
    analysis_path = 'C:/Users/thladnik/TEMP'
    analysis_filename = 'analysis01'

    create_analysis(analysis_path=os.path.join(analysis_path, analysis_filename), data_root=data_root)
