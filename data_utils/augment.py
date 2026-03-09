import librosa
import numpy as np
from audiomentations import Compose, TimeStretch, PitchShift, Shift, Gain, AddBackgroundNoise, AddGaussianNoise

def faint_start(sequence, samplerate, shift_seconds=0.2, scale=0.1, noise_level=0.002):
    # Make the first shift_seconds faint by scale factor and add noise
    end = int(shift_seconds*samplerate)
    scale = [scale if i<end else 1 for i in range(len(sequence))]
    additive_noise = [noise_level*np.random.normal() if i<end else 0 for i in range(len(sequence))]
    return sequence*scale + additive_noise

def clip_start(sequence, samplerate, shift_seconds=0.3, noise_level=0.002):
    # Clips the first shift_seconds while adding noise at the beginning
    end = int(shift_seconds*samplerate)
    scale = [1 if i<end else 0 for i in range(len(sequence))]
    additive_noise = [0 if i<end else noise_level*np.random.normal() for i in range(len(sequence))]
    return np.roll(sequence*scale+additive_noise, -end, axis=0)


# These data augmentation will be performed all together (code written by Master students)
def add_noise(sequence, noise_level=0.02):
    # Adds Gaussian noise to the envelope sequence.
    return sequence + noise_level * np.random.randn(*sequence.shape)

def time_shift(sequence, samplerate, max_shift_seconds=0.2):
    #Circularly shifts the audio envelope left or right by up to max_shift_seconds
    shift_max = int(samplerate*max_shift_seconds)
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(sequence, shift, axis=0)

def random_scaling(sequence, scale_range=(0.8, 1.2)):
    #Randomly scales the amplitude of the envelope.
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return sequence * scale

def augment_cascade(audio, samplerate, with_noise=True): 
    augmented_audio = random_scaling(audio)
    augmented_audio = time_shift(augmented_audio, samplerate) # doing time_shift before time_stretch because I'm defining shift_max in seconds
    if with_noise: 
        augmented_audio = add_noise(augmented_audio)
    return augmented_audio

# Taken from Ilaria Morandi's code used for https://doi.org/10.64898/2025.12.19.695494 

def augment_distances(fname,samples, 
                      background_noise_folder="Data/Yell_ID_Set/Background_noise/", 
                      snr_levels = [-15, -10, -5, 0, 5, 10, 15, 20, 25, 30], sr=20480):
    # Taken from Ilaria's code
    background_noise_transforms = [
        AddBackgroundNoise(
            sounds_path=background_noise_folder,
            min_snr_db=snr,
            max_snr_db=snr,
            p=1.0  # always apply
        ) for snr in snr_levels
    ]

    other_augment = Compose([
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.4),
        PitchShift(min_semitones=-0.2, max_semitones=0.2, p=0.4),
        Shift(min_shift=-0.3, max_shift=0.3, shift_unit="seconds", rollover=False, p=0.5),
    ])

    augmented_samples = []
    aug_names=['aug{'+str(snr)+"}"+fname for snr in snr_levels]
    for i in range(len(snr_levels)):
            #temp_aug = other_augment(samples=samples, sample_rate=sr)
            final_aug = background_noise_transforms[i](samples=samples, sample_rate=sr)
            augmented_samples.append(final_aug)

    return np.array(augmented_samples), aug_names