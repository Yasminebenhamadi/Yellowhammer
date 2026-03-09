import math
import numpy as np
import pandas as pd
import data_utils.dataset as dataset
import data_utils.augment as augment

GOERTZEL_THRESHOLD_BUFFER_LENGTH = 16384

def generate_hamming_values(N):
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
    return hamming, np.mean(hamming)


def apply_goertzel_filter(samples, sample_rate, freq, N):
    
    hamming_values, hamming_mean = generate_hamming_values(N)
    c = 2.0 * np.cos(2.0 * np.pi * freq / sample_rate)
    
    max_amp = 1.
    maximum = N * max_amp * hamming_mean / 2.0
    scaler = 1.0 / maximum

    d1, d2 = 0.0, 0.0
    output = []

    for i, x in enumerate(samples):
        y = hamming_values[i % N] * x + c * d1 - d2
        d2, d1 = d1, y

        # At the end of each window
        if i % N == N - 1:
            magnitude = (d1 * d1) + (d2 * d2) - c * d1 * d2
            goertzel_value = 0 if magnitude < 0 else np.sqrt(magnitude)
            output.append(min(goertzel_value * scaler, 1.0))
            d1, d2 = 0.0, 0.0

    return np.array(output)

def apply_goertzel_threshold(goertzel_values, threshold, window_length, min_trigger_duration_samples):
    # Convert minimum trigger duration into number of buffers
    min_trigger_duration_buffers = math.ceil(
        min_trigger_duration_samples / GOERTZEL_THRESHOLD_BUFFER_LENGTH
    )

    trigger_duration = 0
    above_threshold = False
    index = 0
    thresholded_value_count = 0

    goertzel_buffer_length = GOERTZEL_THRESHOLD_BUFFER_LENGTH / window_length

    output = []

    while index < len(goertzel_values):
        limit = min(len(goertzel_values), int(index + goertzel_buffer_length))

        while index < limit:
            if goertzel_values[index] > threshold:
                above_threshold = True
                trigger_duration = min_trigger_duration_buffers
            index += 1

        output.append(above_threshold)

        if above_threshold:
            thresholded_value_count += 1
            if trigger_duration > 1:
                trigger_duration -= 1
            else:
                above_threshold = False

    thresholded_value_count *= GOERTZEL_THRESHOLD_BUFFER_LENGTH
    thresholded_value_count = min(
        thresholded_value_count, len(goertzel_values) * window_length
    )

    return output, thresholded_value_count

def goertzel_scores(goertzel_values, window_length):
    goertzel_buffer_length = GOERTZEL_THRESHOLD_BUFFER_LENGTH / window_length

    nb_goertzel_triggers=math.ceil(len(goertzel_values)/goertzel_buffer_length)
    
    index=0
    scores = []
    for trigger in range(nb_goertzel_triggers):
        limit = min(len(goertzel_values), int(index + goertzel_buffer_length)) 
        trigger_values = goertzel_values[index:limit]
        scores.append(np.max(trigger_values))
        index = limit
    return scores

def goertzel_inference(X, sr, center_freq = 5500, window_len = 32):
    y_scores = []
    for sample in X:
        goertzel_values = apply_goertzel_filter(sample.flatten(), sr, center_freq, window_len)
        score = goertzel_scores(goertzel_values, window_len)
        y_scores.append(score)
    return np.array(y_scores)