import os
import pickle
import time
from typing import Any, TYPE_CHECKING

import h5py


def write(analysis, key: str, value: Any, data_path: str):

    # Decode data path
    file_type, *file_info = data_path.split(':')
    if file_type == 'hdf5':
        path, key = file_info
    else:
        path, = file_info

    # Write data to file
    if file_type == 'hdf5':
        pending = True
        start = time.perf_counter()
        while pending:
            try:
                with h5py.File(os.path.join(analysis.analysis_path, path), 'a') as f:
                    if key not in f:
                        f.create_dataset(key, data=value,
                                         compression=analysis.compression,
                                         compression_opts=analysis.compression_opts)
                    else:
                        if value.shape != f[key].shape:
                            del f[key]
                            f.create_dataset(key, data=value,
                                             compression=analysis.compression,
                                             compression_opts=analysis.compression_opts)
                        else:
                            f[key][:] = value

            except Exception as _exc:
                if (time.perf_counter() - start) > analysis.write_timeout:
                    import traceback

                    raise TimeoutError(f'Failed to write attribute {key} to {key} '
                                       f'// Traceback: {traceback.format_exc()}')
                else:
                    time.sleep(10 ** -6)
            else:
                pending = False

    # TODO: alternative to pickle dumps? Writing arbitrary raw binary data to HDF5 seems difficult
    # Dump all other types as binary strings
    else:
        with open(os.path.join(analysis.analysis_path, path), 'wb') as f:
            pickle.dump(value, f)


def read(analysis, data_path: str) -> Any:

    # Load and return referenced dataset
    file_type, *file_info = data_path.split(':')

    if file_type == 'hdf5':
        file_path, key = file_info
        with h5py.File(os.path.join(analysis.analysis_path, file_path), 'r') as f:
            return f[key][:]

    elif file_type == 'pkl':
        file_path, = file_info
        with open(os.path.join(analysis.analysis_path, file_path), 'rb') as f2:
            return pickle.load(f2)

    else:
        raise Exception(f'Unknown file type {file_type} for path {data_path}')

def delete(*args):
    pass