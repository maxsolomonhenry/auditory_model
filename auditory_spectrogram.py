"""
Audi: An auditory spectrogram for Python ported from NSL Tools:
    Ru, Powen. 2001. “Multiscale multirate spectro-temporal auditory model.”
    http://www.isr.umd.edu/~speech/nsltools.tar.gz

"""

# TODO:     make more readable --
#               (1) clean up Matlab index-stuffs
#               (2) transpose calculations.

import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.io import wavfile, loadmat

x = loadmat('aud24.mat')
COCHBA = x['COCHBA']


def auditory_spectrogram(x, params, verbose=False):
    [L, M] = COCHBA.shape

    octave_shift = params['octave_shift']
    nonlinear_factor = params['nonlinear_factor']
    frame_length = int(round(params['frame_length'] * 2 ** (4 + octave_shift)))

    if params['decay_factor']:
        alpha = np.exp(-1/(params['decay_factor'] * 2 ** (4 + octave_shift)))
    else:
        alpha = 0

    haircell_tc = 0.5
    beta = np.exp(-1/(haircell_tc * 2 ** (4 + octave_shift)))

    num_frames = int(np.ceil(len(x) / frame_length))
    x = pad_to(x, num_frames * frame_length)
    X = np.zeros([num_frames, M-1])

    # Highest channel is processed independently
    filter_order = int(np.real(COCHBA[0, M-1]))
    b = np.real(COCHBA[np.arange(1, filter_order + 2), M-1])
    a = np.imag(COCHBA[np.arange(1, filter_order + 2), M-1])

    y = scipy.signal.lfilter(b, a, x)
    y = apply_nonlinearity(y, nonlinear_factor)
    y = scipy.signal.lfilter([1], [1, -beta], y)


    y_h = copy.deepcopy(y)

    if verbose:
        t0 = time.time()

    # Process all other channels
    for channel in range(M - 1, 1, -1):
        filter_order = int(np.real(COCHBA[0, channel-1]))
        b = np.real(COCHBA[np.arange(1, filter_order + 2), channel-1])
        a = np.imag(COCHBA[np.arange(1, filter_order + 2), channel-1])

        y = scipy.signal.lfilter(b, a, x)
        y = apply_nonlinearity(y, nonlinear_factor)
        y = scipy.signal.lfilter([1], [1, -beta], y)

        temp = y - y_h
        y_h = copy.deepcopy(y)
        y = np.maximum(temp, 0)

        if alpha:
            y = scipy.signal.lfilter([1], [1, -alpha], y)
            X[:, channel - 1] = y[range(0, num_frames * frame_length, frame_length)]
        elif frame_length == 1:
            X[:, channel - 1] = y
        # else:
        #     X[:, channel] = y[frame_length]

    if verbose:
        delta_t = time.time() - t0
        print("Took {}s.".format(delta_t))

    return X.T


def pad_to(x, length):
    assert length > len(x), "Desired length must be longer than input array."
    return np.pad(x, (0, length - len(x)))


def apply_nonlinearity(in_, nonlinear_factor):
    if nonlinear_factor > 0:
        out_ = 1/(1 + np.exp(-in_/nonlinear_factor))
    elif nonlinear_factor == 0:
        out_ = (in_ > 0)
    elif nonlinear_factor == -1:
        out_ = np.maximum(0, in_)
    return out_


if __name__ == '__main__':
    # Example.

    sr, audio = wavfile.read('./audio/Tpt1_vib50.wav')
    audio = audio / 2**15

    # Resample to 8kHz.
    seconds = len(audio) / sr
    audio = scipy.signal.resample(audio, int(seconds * 8000))

    params = {'octave_shift': -1,
              'nonlinear_factor': 0.1,
              'frame_length': 4,
              'decay_factor': 4
              }

    X = auditory(audio, params, verbose=True)
    plt.imshow(X, origin='lower', aspect='auto')
    plt.show()