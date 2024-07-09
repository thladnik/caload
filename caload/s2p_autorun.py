"""Module runs the suite2p pipeline on all recording subfolders in a given
root directory and performs a registration of all recordings to a zstack,
if a zstack tif file is located in the root directory. Zstack registration
results are saved to zstack_corrs.npy (phase correlation to planes)
and zstack_xy.npy (x/y pixel shift within planes)."""

__all__ = ['run_pipline', 'phase_correlations',
           'ops_2p_jf7_1p6mag', 'ops_2p_jf7_1p8mag', 'ops_2p_jf7_2p0mag']
__author__ = 'Tim Hladnik'
__version__ = '0.0.1'

import os
import pprint
from pathlib import Path
from typing import Any, Dict

import numpy as np
import suite2p
from tifffile import tifffile

ops_2p_jf7_1p6mag = {
    'nonrigid': True,
    'diameter': 12,
    'denoise': True,
    'anatomical_only': 2,
    'neuropil_extract': False,
    'tau': 1.61,
    'fs': 2.18
}

ops_2p_jf7_1p8mag = {
    'nonrigid': True,
    'diameter': 12,
    'denoise': True,
    'anatomical_only': 2,
    'neuropil_extract': False,
    'tau': 1.61,
    'fs': 2.18
}

ops_2p_jf7_2p0mag = {
    'nonrigid': True,
    'diameter': 14,
    'denoise': True,
    'anatomical_only': 2,
    'neuropil_extract': False,
    'tau': 1.61,
    'fs': 2.18
}


def phase_correlations(ref: np.ndarray, im: np.ndarray) -> np.ndarray:
    """Phase correlation calculation
    after: https://github.com/michaelting/Phase_Correlation/blob/master/phase_corr.py

    Parameters
    ----------
    ref : array
        Array of shape (N, M)
    im : array
        Array of shape (N, M)

    Returns
    -------
    array
        Array of same size as ref, containing the phase correlations for the
        corresponding index shift.
    """

    conj = np.ma.conjugate(np.fft.fft2(im))
    r = np.fft.fft2(ref) * conj
    r /= np.absolute(r)
    return np.fft.ifft2(r).real


def run_pipline(path: str, custom_ops: Dict[str, Any] = None) -> None:
    """Run the suite2p pipeline on all subfolders in the root directory
    specified by `path`, if they contain a tif file (calcium recording).

    If the root directory contains a tif file with the `zstack` substring,
    a zstack registration of suite2p's registered `meanImg` against the zstack
    is performed.

    Expected root directory structure:

        path
        +-- zstack_tif_file.tif (optional)
        +-- recording_folder01
        |   +-- tif_file01.tif
        +-- recording_folder02
        |   +-- tif_file02.tif
        +-- recording_folder03
        |   +-- tif_file03.tif
        ...
        +-- recording_foldern
        |   +-- tif_filen.tif

    Parameters
    ----------
    path : string
        Path to the root folder which should be scanned for recording subfolders
    custom_ops: Dict[str, Any]
        Dictionary containing all suite2p processing ops that should be used
        to overwrite s2p's `default_ops`

    """

    path = Path(path).as_posix()

    # Set custom_ops to empty dictionary
    if custom_ops is None:
        custom_ops = {}

    # Check if root is valid directory
    if not os.path.exists(path) and os.path.isdir(path):
        raise FileNotFoundError('Root path does not exist or is a file')

    # Scan root folder to find zstack
    filenames = [f for f in os.listdir(path) if os.path.isfile(f'{path}{f}')]
    zstack_filenames = [f for f in filenames if 'zstack' in f]
    zstack = None
    if len(zstack_filenames) > 0:
        print(f'Found zstack: {zstack_filenames[0]}')

        # Load stack
        zstack = tifffile.imread(f'{path}{zstack_filenames[0]}')

        if len(zstack_filenames) > 1:
            print(f'WARNING: multiple zstack files found. Using first one: {zstack_filenames[0]}')
    else:
        print('WARNING: no zstack found')

    # Prepare pipeline ops
    print('Search for recording folders')
    ops_list = []
    for fn in os.listdir(path):

        recording_path = os.path.join(path, fn)
        # Skip any files
        if os.path.isfile(recording_path):
            continue

        # Set ops
        ops = suite2p.default_ops()
        # Overwrite with custom ops
        ops.update(custom_ops)
        # Set data_path to tif filepath
        ops['data_path'] = [recording_path]
        # Add to list
        ops_list.append(ops)

    print(f'Found {len(ops_list)} recordings to process:')
    pprint.pprint([ops['data_path'][0] for ops in ops_list])

    # Run pipeline
    print('Run pipelines')
    for ops in ops_list:

        recording_path = ops["data_path"][0]

        # Run s2p pipeline
        print('#' * 32)
        print(f'Run s2p for data_path {recording_path}')
        output_ops = suite2p.run_s2p(ops=ops)

        # Output to simplified mat file:
        # stat = np.load(os.path.join(recording_path, 'suite2p', 'plane0', 'stat.npy'))
        # F = np.load(os.path.join(recording_path, 'suite2p', 'plane0', 'F.npy'))
        #
        # matpath = os.path.join(recording_path, 'Fall.mat')
        # scipy.io.savemat(matpath, {'stat': stat, 'ops': output_ops, 'F': F})

        # If no zstack was found, this is it
        if zstack is None:
            continue

        # Do zstack registration if file exists
        print('Run zstack registration')

        ref = output_ops['meanImg'][:]
        # Determine padding and make sure it is divisible by 2
        padding = ref.shape[0] / 4
        padding = int(padding // 2 * 2)

        # Pad reference on all sides
        ref_im = np.pad(ref, (padding // 2, padding // 2))

        corrs = []
        xy = []
        for im in zstack:
            image = np.pad(im, (0, padding))

            corrimg = phase_correlations(ref_im, image)
            maxcorr = corrimg.max()
            y, x = np.unravel_index(corrimg.argmax(), corrimg.shape)

            x -= padding // 2
            y -= padding // 2

            corrs.append(maxcorr)
            xy.append([x, y])

        print('Save registration results')
        # Save
        np.save(f'{recording_path}zstack_corrs.npy', np.array(corrs))
        np.save(f'{recording_path}zstack_xy.npy', np.array(xy))


if __name__ == '__main__':
    # Root path to scan for recording folders (which contain the tif files)
    root_folder = './'

    # Run pipeline
    run_pipline(root_folder, custom_ops=ops_2p_jf7_1p6mag)
