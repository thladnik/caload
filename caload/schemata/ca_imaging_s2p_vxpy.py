"""Structure and functions for analysis of imaging experiments done with vxPy.

Expected raw data folder/file structure:

data_root
|   + animal_id_01
|   +-- zstack_for_animal_id_01.tif (optional)
|   +-- recording_id_01
|   |   +-- rec_file_01.tif
|   ...
|   +-- recording_id_n
|   |   +-- rec_file_n.tif
|   + animal_id_02
|   +-- zstack_for_animal_id_02.tif (optional)
|   +-- recording_id_01
|   |   +-- rec_file_01.tif
|   ...
|   +-- recording_id_n
|   |   +-- rec_file_n.tif
...

Output data hierarchy:

Animal
+-- Recording
    +-- Roi
    +-- Phase

"""
import os
from pathlib import Path
from typing import List, Tuple, Type, Union

import h5py
import numpy as np
import scipy
import yaml
from tifffile import tifffile
from tqdm import tqdm

import caload
from caload.entities import *

__all__ = ['Animal', 'Recording', 'Roi', 'Phase', 'digest_folder', 'schema']


class Animal(Entity):

    @property
    def recordings(self) -> EntityCollection:
        return self.analysis.get(Recording, animal_id=self.id)

    @property
    def rois(self) -> EntityCollection:
        return self.analysis.get(Roi, animal_id=self.id)

    @property
    def phases(self) -> EntityCollection:
        return self.analysis.get(Phase, animal_id=self.id)


class Recording(Entity):

    parent_type = Animal
    animal: Animal

    def load(self):
        self.animal = Animal(analysis=self.analysis, row=self.row.parent)

    # @property
    # def animal(self) -> Animal:
    #     return Animal(analysis=self.analysis, row=self._row.parent)

    @property
    def rois(self) -> EntityCollection:
        return self.analysis.get(Roi, animal_id=self.animal.id, rec_id=self.id)

    @property
    def phases(self) -> EntityCollection:
        return self.analysis.get(Phase, animal_id=self.animal.id, rec_id=self.id)


class Roi(Entity):

    parent_type = Recording
    animal: Animal
    recording: Recording

    def load(self):
        self.recording = Recording(analysis=self.analysis, row=self.row.parent)
        self.animal = Animal(analysis=self.analysis, row=self.row.parent.parent)

    # @property
    # def animal(self) -> Animal:
    #     return Animal(analysis=self.analysis, row=self.row.parent.parent)

    # @property
    # def recording(self) -> Recording:
    #     return Recording(analysis=self.analysis, row=self.row.parent)


class Phase(Entity):

    parent_type = Recording
    animal: Animal
    recording: Recording

    def load(self):
        self.recording = Recording(analysis=self.analysis, row=self.row.parent)
        self.animal = Animal(analysis=self.analysis, row=self.row.parent.parent)

    # @property
    # def animal(self) -> Animal:
    #     return Animal(analysis=self.analysis, row=self.row.parent.parent)
    #
    # @property
    # def recording(self) -> Recording:
    #     return Recording(analysis=self.analysis, row=self.row.parent)


schema = [Animal, Recording, Roi, Phase]


def digest_folder(analysis: caload.analysis.Analysis):

    # Scan for data folders
    folder_list = scan_folder(analysis.config['data_root'], [])

    print(f'Process folders: {folder_list}')
    for recording_path in folder_list:

        # recording_path = Path(recording_path).as_posix()
        print(f'Recording folder {recording_path}')

        # Add animal
        animal = get_animal(analysis, recording_path)
        analysis.session.commit()

        recording = get_recording(animal, recording_path)

        # Load s2p processed data
        s2p_path = os.path.join(recording_path, 'suite2p', 'plane0')

        # Load suite2p's analysis options
        print('Include suite2p ops')
        ops = np.load(os.path.join(s2p_path, 'ops.npy'), allow_pickle=True).item()
        unravel_dict(ops, recording, 's2p')

        print('Calculate frame timing of signal')
        with h5py.File(os.path.join(recording_path, 'Io.hdf5'), 'r') as io_file:

            mirror_position = np.squeeze(io_file['ai_y_mirror_in'])[:]
            mirror_time = np.squeeze(io_file['ai_y_mirror_in_time'])[:]

            # Calculate frame timing
            frame_idcs, frame_times = calculate_ca_frame_times(mirror_position, mirror_time)

            record_group_ids = io_file['__record_group_id'][:].squeeze()
            record_group_ids_time = io_file['__time'][:].squeeze()

            ca_rec_group_id_fun = scipy.interpolate.interp1d(record_group_ids_time, record_group_ids, kind='nearest')

        print('Load ROI data')
        fluorescence = np.load(os.path.join(s2p_path, 'F.npy'), allow_pickle=True)
        spikes_all = np.load(os.path.join(s2p_path, 'spks.npy'), allow_pickle=True)
        roi_stats_all = np.load(os.path.join(s2p_path, 'stat.npy'), allow_pickle=True)
        # In some suite2p versions the iscell file may be missing?
        try:
            iscell_all = np.load(os.path.join(s2p_path, 'iscell.npy'), allow_pickle=True)
        except:
            iscell_all = None

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
        record_group_ids = ca_rec_group_id_fun(frame_times)
        recording['record_group_ids'] = record_group_ids

        # Commit recording
        analysis.session.commit()

        # Add suite2p's analysis ROI stats
        print('Add ROI stats and signals')
        rois = recording.add_child_entity(Roi, entity_id=[f'roi_{roi_id}' for roi_id in range(fluorescence.shape[0])])
        for roi_id, roi in tqdm(enumerate(rois)):
            # Create ROI
            # roi = recording.add_child_entity(Roi, entity_id=f'roi_{roi_id}')
            roi['animal_id'] = animal.id
            roi['rec_id'] = recording.id
            roi['roi_id'] = roi_id

            roi_stats = roi_stats_all[roi_id]

            # Write ROI stats
            roi.update({f's2p/{k}': v for k, v in roi_stats.items()})

            # Write data
            fluores = fluorescence[roi_id]
            spikes = spikes_all[roi_id]
            roi['fluorescence'] = fluores
            roi['spikes'] = spikes

            if iscell_all is not None:
                iscell = iscell_all[roi_id]
                roi['iscell'] = iscell

        # Commit rois
        analysis.session.commit()

        print('Add display phase data')
        with h5py.File(os.path.join(recording_path, 'Display.hdf5'), 'r') as disp_file:

            # Get attributes
            recording.update({f'display/attrs/{k}': v for k, v in disp_file.attrs.items()})

            for key1, member1 in tqdm(disp_file.items()):

                # If dataset, write to file
                if isinstance(member1, h5py.Dataset):
                    recording[f'display/{key1}'] = np.squeeze(member1[:])
                    continue

                # Otherwise it's a group -> keep going

                # Add phase
                if 'phase' in key1:
                    phase_id = key1

                    phase_id_int = int(key1.replace('phase', ''))
                    phase = recording.add_child_entity(Phase, entity_id=phase_id)
                    phase['animal_id'] = animal.id
                    phase['rec_id'] = recording.id
                    phase['phase_id'] = phase_id
                    phase['phase_id_int'] = phase_id_int

                    # Add calcium start/end indices
                    in_phase_idcs = np.where(record_group_ids == phase_id_int)[0]
                    start_index = np.argmin(np.abs(frame_times - frame_times[in_phase_idcs[0]]))
                    end_index = np.argmin(np.abs(frame_times - frame_times[in_phase_idcs[-1]]))
                    phase['ca_start_index'] = start_index
                    phase['ca_end_index'] = end_index

                    # Write attributes
                    phase.update({k: v for k, v in member1.attrs.items()})

                    # Write datasets
                    for key2, member2 in member1.items():
                        if isinstance(member2, h5py.Dataset):
                            phase[key2] = np.squeeze(member2[:])

                # Add other data
                else:
                    # Write attributes
                    recording.update({f'display/{key1}/{k}': v for k, v in member1.attrs.items()})

                    # Get datasets
                    for key2, member2 in member1.items():
                        if isinstance(member2, h5py.Dataset):
                            recording[f'display/{key1}/{key2}'] = np.squeeze(member2[:])

        # Commit phases and display data
        analysis.session.commit()


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


def get_animal(analysis: caload.analysis.Analysis, path: str) -> Entity:

    # Create animal
    animal_id = Path(path).as_posix().split('/')[-2]

    animal_collection = analysis.get(Animal, f'animal_id == "{animal_id}"')

    if len(animal_collection) > 0:
        if len(animal_collection) > 1:
            raise Exception(f'Got {len(animal_collection)} animals with ID {animal_id}')

        return animal_collection[0]

    animal = analysis.add_entity(Animal, animal_id)
    animal['animal_id'] = animal_id

    # Get path to animal folder
    animal_path = '/'.join(Path(path).as_posix().split('/')[:-1])

    # Search for zstacks
    zstack_names = []
    for fn in os.listdir(animal_path):
        path = os.path.join(path, fn)
        if os.path.isdir(path):
            continue
        if 'zstack' in fn:
            if fn.lower().endswith(('.tif', '.tiff')):
                zstack_names.append(fn)

    if len(zstack_names) > 0:
        if len(zstack_names) > 1:
            print(f'WARNING: multiple zstacks detected, using {zstack_names[0]}')

        print(f'Add zstack {zstack_names[0]}')

        animal['zstack_fn'] = zstack_names[0]
        animal['zstack'] = tifffile.imread(os.path.join(animal_path, zstack_names[0]))

    # Add metadata
    add_metadata(animal, animal_path)

    # Commit animal
    analysis.session.commit()

    return animal


def get_recording(animal: Entity, path: str) -> Entity:
    # Create debug folder
    debug_folder_path = os.path.join(path, 'debug')
    if not os.path.exists(debug_folder_path):
        os.mkdir(debug_folder_path)

    # Get recording
    rec_id = Path(path).as_posix().split('/')[-1]
    expr = f'animal_id == "{animal.id}" AND rec_id == "{rec_id}"'
    recording_collection = animal.analysis.get(Recording, expr)
    # Add recording
    if len(recording_collection) > 0:
        if len(recording_collection) > 1:
            raise Exception(f'Got {len(recording_collection)} recordings with '
                            f'animal_id == "{animal.id}" and rec_id == "{rec_id}"')

        return recording_collection[0]

    recording = animal.add_child_entity(Recording, rec_id)
    recording['animal_id'] = animal.id
    recording['rec_id'] = rec_id

    # Add metadata
    add_metadata(recording, path)

    return recording


def add_metadata(entity: caload.entities.Entity, folder_path: str):
    """Function searches for and returns metadata on a given folder path

    Function scans the `folder_path` for metadata yaml files (ending in `meta.yaml`)
    and returns a dictionary containing their contents
    """

    meta_files = [f for f in os.listdir(folder_path) if f.endswith('metadata.yaml')]

    print(f'Found {len(meta_files)} metadata files in {folder_path}.')

    metadata = {}
    for f in meta_files:
        with open(os.path.join(folder_path, f), 'r') as stream:
            try:
                metadata.update(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)

    # Add metadata
    unravel_dict(metadata, entity, 'metadata')


def unravel_dict(dict_data: dict, entity: caload.entities.Entity, path: str):
    for key, item in dict_data.items():
        if isinstance(item, dict):
            unravel_dict(item, entity, f'{path}/{key}')
            continue
        entity[f'{path}/{key}'] = item


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


if __name__ == '__main__':
    pass
