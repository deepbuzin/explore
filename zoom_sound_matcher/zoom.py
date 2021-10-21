import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pyAudioAnalysis import audioSegmentation as aS
import glob
import os
import argparse


def _load(rec_path, sr=44100):
    print('loading audio from {}'.format(rec_path))
    all_y, _ = librosa.load(os.path.join(rec_path, 'audio_only.m4a'), sr=sr)
    ys = list()

    for path in glob.glob(os.path.join(rec_path, '*', '*.m4a')):
        y, _ = librosa.load(path, sr=sr)
        ys.append(y)

    return all_y, ys


def _detect_activity_coarse(y, offset, t=0.5, sr=44100):
    print('detecting voice activity...')
    segment_size = int(t * sr)  # segment size in samples

    segments = np.array([y[x:x + segment_size] for x in np.arange(0, len(y), segment_size)])
    energies = [(s ** 2).sum() / len(s) for s in segments]

    thresh = 0.5 * np.median(energies)
    live_idx = (np.where(energies > thresh)[0])

    return [(i * t + offset, t) for i in live_idx]


def _detect_activity(y, offset, sr=44100):
    print('detecting voice activity...')
    segments = aS.silence_removal(y, sr, 0.020, 0.020, smooth_window=1.0, weight=0.3, plot=False)
    return [(s[0] + offset, s[1] - s[0]) for s in segments]


def _find_offset(all_y, single_y, window_t=10, sr=44100):
    window = window_t * sr
    sample = single_y[:window]
    z = signal.correlate(all_y, sample)

    peak = np.argmax(np.abs(z))
    offset = (peak - window + 1) / sr

    print('sample offset {}'.format(offset))
    return offset


def _draw(live_ts, limit_speakers=20):
    fig, ax = plt.subplots()

    ticks = np.linspace(5, 5 + len(live_ts) - 1, len(live_ts))
    labels = ['speaker_{}'.format(i) for i in range(len(live_ts))]

    for i, t in enumerate(live_ts):
        ax.broken_barh(t, (i + 5, 1))

    ax.set_ylim(0, limit_speakers + 5)
    ax.set_xlabel('time, s')
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process zoom recordings')
    parser.add_argument('--path', metavar='P', type=str, default='data/meeting/',
                        help='path to the recording dir')
    parser.add_argument('--sample_rate', type=int, default=44100,
                        help='sample rate Hz')
    parser.add_argument('--cc_window', type=int, default=10,
                        help='cross-correlation window size for calculating the offsets, s')
    parser.add_argument('--speaker_limit', type=int, default=20,
                        help='max number of speakers in plot')

    args = parser.parse_args()

    all_y, ys = _load(args.path, sr=args.sample_rate)

    offsets = [_find_offset(all_y, y, window_t=args.cc_window, sr=args.sample_rate) for y in ys]
    live_ts = [_detect_activity(y, offset=offset, sr=args.sample_rate) for y, offset in zip(ys, offsets)]

    _draw(live_ts, limit_speakers=args.speaker_limit)
