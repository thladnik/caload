import numpy as np
import matplotlib.pyplot as plt


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
