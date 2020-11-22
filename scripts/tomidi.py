import numpy as np
from working import writemidi


def load_npy_data(npy_data):
    npy_A = np.load(npy_data[0]) * 1.0  # 64 * 84 * 1
    npy_B = np.load(npy_data[1]) * 1.0  # 64 * 84 * 1
    npy_AB = np.concatenate(
        (
            npy_A.reshape(npy_A.shape[0], npy_A.shape[1], 1),
            npy_B.reshape(npy_B.shape[0], npy_B.shape[1], 1),
        ),
        axis=2,
    )  # 64 * 84 * 2
    return npy_AB


def save_midis(bars, file_path, tempo=80.0):
    padded_bars = np.concatenate(
        (
            np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])),
            bars,
            np.zeros((bars.shape[0], bars.shape[1], 20, bars.shape[3])),
        ),
        axis=2,
    )
    padded_bars = padded_bars.reshape(
        -1, 64, padded_bars.shape[2], padded_bars.shape[3]
    )
    padded_bars_list = []
    for ch_idx in range(padded_bars.shape[3]):
        padded_bars_list.append(
            padded_bars[:, :, :, ch_idx].reshape(
                padded_bars.shape[0], padded_bars.shape[1], padded_bars.shape[2]
            )
        )

    writemidi.write_piano_rolls_to_midi(
        piano_rolls=padded_bars_list,
        program_nums=[0],
        is_drum=[False],
        filename=file_path,
        tempo=tempo,
        beat_resolution=4,
    )


if __name__ == "__main__":
    data = np.load("./JC_J/test/jazz_piano_test_1.npy") * 1.0
    data = data.reshape(1, data.shape[0], data.shape[1], 1)
    save_midis(data, "uwu.mid")
