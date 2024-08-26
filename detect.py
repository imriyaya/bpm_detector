import array
import math
import wave

import numpy
import pywt
from matplotlib import pyplot as plt
from scipy import signal


def read_wav(filename):
    try:
        wavefile = wave.open(filename, 'rb')
    except IOError as e:
        print(e)
        return None

    nframes = wavefile.getnframes()
    assert nframes > 0

    framerate = wavefile.getframerate()
    assert framerate > 0

    frames = list(array.array('i', wavefile.readframes(nframes)))

    try:
        assert nframes == len(frames)
    except AssertionError:
        print(nframes, 'not equal to', len(frames))

    return frames, framerate


def peak_detection(data):
    max_val = numpy.max(abs(data))
    peak_ndx = numpy.where(data == max_val)
    if len(peak_ndx[0]) == 0:
        peak_ndx = numpy.where(data == -max_val)
    return peak_ndx

def bpm_detector(data, framerate):
    cA = []
    cD = []
    correl = []
    cD_sum = []
    levels = 4
    max_decimation = 2 ** (levels - 1)
    min_ndx = math.floor(60.0 / 220 * (framerate / max_decimation))
    max_ndx = math.floor(60.0 / 40 * (framerate / max_decimation))

    for loop in range(0, levels):
        cD = []
        # 1) DWT
        if loop == 0:
            [cA, cD] = pywt.dwt(data, "db4")
            cD_minlen = len(cD) / max_decimation + 1
            cD_sum = numpy.zeros(math.floor(cD_minlen))
        else:
            [cA, cD] = pywt.dwt(cA, "db4")

        # 2) Filter
        cD = signal.lfilter([0.01], [1 - 0.99], cD)

        # 4) Subtract out the mean.

        # 5) Decimate for reconstruction later.
        cD = abs(cD[:: (2 ** (levels - loop - 1))])
        cD = cD - numpy.mean(cD)

        # 6) Recombine the signal before ACF
        #    Essentially, each level the detail coefs (i.e. the HPF values) are concatenated to the beginning of the array
        cD_sum = cD[0 : math.floor(cD_minlen)] + cD_sum

    if [b for b in cA if b != 0.0] == []:
        print("No audio data for sample, skipping...")
        return None, None

    # Adding in the approximate data as well...
    cA = signal.lfilter([0.01], [1 - 0.99], cA)
    cA = abs(cA)
    cA = cA - numpy.mean(cA)
    cD_sum = cA[0 : math.floor(cD_minlen)] + cD_sum

    # ACF
    correl = numpy.correlate(cD_sum, cD_sum, "full")

    midpoint = math.floor(len(correl) / 2)
    correl_midpoint_tmp = correl[midpoint:]
    peak_ndx = peak_detection(correl_midpoint_tmp[min_ndx:max_ndx])
    if len(peak_ndx) > 1:
        print("No audio data for sample, skipping...")
        return None, None

    peak_ndx_adjusted = peak_ndx[0] + min_ndx
    bpm = 60.0 / peak_ndx_adjusted * (framerate / max_decimation)
    print(bpm)
    return bpm, correl


if __name__ == "__main__":
    samps, fs = read_wav('wav.wav')
    data = []
    correl = []
    bpm = 0
    n = 0
    nsamps = len(samps)
    window_samps = int(3 * fs)
    samps_ndx = 0  # First sample in window_ndx
    max_window_ndx = math.floor(nsamps / window_samps)
    bpms = numpy.zeros(max_window_ndx)

    # Iterate through all windows
    for window_ndx in range(0, max_window_ndx):

        # Get a new set of samples
        data = samps[samps_ndx : samps_ndx + window_samps]
        if not ((len(data) % window_samps) == 0):
            raise AssertionError(str(len(data)))

        bpm, correl_temp = bpm_detector(data, fs)
        if bpm is None:
            continue
        bpms[window_ndx] = bpm
        correl = correl_temp

        # Iterate at the end of the loop
        samps_ndx = samps_ndx + window_samps

        # Counter for debug...
        n = n + 1

    bpm = numpy.median(bpms)
    print("Completed!  Estimated Beats Per Minute:", bpm)

    n = range(0, len(correl))
    plt.plot(n, abs(correl))
    plt.show(block=True)