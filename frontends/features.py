import os
import librosa
import numpy as np
import pandas as pd
import cmsisdsp as dsp
from scipy.signal import butter, windows
from numpy.lib.stride_tricks import sliding_window_view


# TODO for ESP

# For AudioMoth
## TODO stft and mels on AM

class MelCMSIS:

    def __init__(self, samplerate, window_len, window_stride, nb_mels, fmin, fmax):
        self.samplerate = samplerate
        self.window_len = window_len
        self.window_stride = window_stride
        self.nb_mels = nb_mels
        self.fmin = fmin
        self.fmax = fmax

        self.hann_window = windows.hann(self.window_len)
        self.fft_istance = dsp.arm_rfft_fast_instance_f32()
        init_STATUS = dsp.arm_rfft_fast_init_f32(self.fft_istance, self.window_len)
        assert(init_STATUS==0)

        self.mel_filter_pos, self.mel_filter_len, self.mel_filter_coefs = self._truncate_mels(samplerate, window_len, nb_mels, fmin, fmax)
    
    @staticmethod
    def _truncate_mels(samplerate, window_len, nb_mels, fmin, fmax):
        '''
        This gives truncated mel filterbanks that are meant to be applied per window
        i.e to a row of size (window_len/2 + 1,)
        '''
        mel_filter= librosa.filters.mel(sr=samplerate, n_fft=window_len, n_mels=nb_mels, fmin=fmin, fmax=fmax)

        mel_filter_pos = []
        mel_filter_len = []
        mel_filter_coefs = []

        for bin_idx, mel_bin in enumerate(mel_filter):

            nonzero = np.where(mel_bin > 0)[0]
            if len(nonzero) == 0:
                continue

            start_idx, end_idx = nonzero[0], nonzero[-1] + 1

            mel_filter_pos.append(start_idx)
            mel_filter_len.append(end_idx-start_idx)
            mel_filter_coefs.extend(mel_bin[start_idx:end_idx])

        assert(len(mel_filter_pos)==nb_mels)
        return mel_filter_pos, mel_filter_len, mel_filter_coefs
    
    def _apply_truncate_mels(self, window):

        '''
        Applies truncated mel filterbanks per window and returns (nb_mels, 1)
        '''

        mel_column = np.zeros((self.nb_mels))

        mel_idx=0
        
        for idx_mel in range(self.nb_mels):
            idx_win_start=self.mel_filter_pos[idx_mel]
            mel_size=self.mel_filter_len[idx_mel]
            trunc_window=window[idx_win_start:idx_win_start+mel_size]

            mel_coefs=self.mel_filter_coefs[mel_idx:mel_idx+mel_size]
            mel_idx=mel_idx+mel_size

            mel_column[idx_mel] = np.dot(trunc_window, mel_coefs)

        return mel_column

    def _process_window(self, window):
        hanned = dsp.arm_mult_f32(window, self.hann_window)

        # arm_rfft_fast_f32 returns packed nfft/2 elements layed out as [real(0), real(N/2), real(1), imag(1) .... real(N/2-1), imag(N/2-1)]
        fft = dsp.arm_rfft_fast_f32(self.fft_istance,
                                    hanned,
                                    0) # ifftFlag=0 for inverse fft
        
        # Expand back to nfft/2 + 1
        cmsis_cmplx_power = np.concatenate([
            [fft[0] * fft[0]],                 # DC
            dsp.arm_cmplx_mag_squared_f32(fft[2:]),
            [fft[1] * fft[1]]                  # Nyquist
        ]) 
        
        mel_spec = self._apply_truncate_mels(cmsis_cmplx_power)
        
        return dsp.arm_vlog_f32(mel_spec)

    def cmsis_log_mel_spectogram(self, audio):
        v = sliding_window_view(audio, self.window_len)[::self.window_stride]
        outs = []
        for row in v:
            outs.append(self._process_window(row))
        return np.array(outs).T


def mel_preprocess(y, sr=20480):
    S = librosa.feature.melspectrogram(y=y, sr=sr, fmin=2000, fmax=10000, n_mels=32,hop_length=320, win_length=512)
    db_S = librosa.power_to_db(S, ref=np.max)
    return db_S

def float_to_fixed_q31(x):
    q31 = (x * (1 << 31)).astype(np.int64)  # use int64 to prevent overflow
    q31 = np.clip(q31, -0x80000000, 0x7FFFFFFF)
    return q31.astype(np.int32) 

def fixed_q31_to_float(x):
    return x / float(1 << 31)

def float_to_fixed_q15(x):
    return np.array(x*32768).astype(np.int16)

def fixed_q15_to_fixed_q31(x):
    return x.astype(int) << 16

# Using CMSIS
def sos_bandpass_filter(lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut /  nyquist
    sos = butter(order, [low, high], btype = 'band', output = 'sos')
    return sos

def sos_lowpass_filter(lowcut, fs, order=6):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    sos = butter(order, low, btype = 'low', analog=False, output = 'sos')
    return sos

def get_cmsis_filter(sos):
    numStages = sos.shape[0]
    new_shape = numStages*(sos.shape[1] - 1)
    pCoeffs=np.reshape(np.hstack((sos[:,:3],-sos[:,4:])), new_shape)
    return numStages, pCoeffs

def filter_cmsisdsp_f32(signal, numStages, pCoeffs):
    # Initialize filter
    biquad_cascade_instance = dsp.arm_biquad_cascade_df2T_instance_f32()
    status = dsp.arm_biquad_cascade_df2T_init_f32(biquad_cascade_instance, numStages, pCoeffs, np.zeros(numStages*4))
    # Apply filter
    signal_filtered = dsp.arm_biquad_cascade_df2T_f32(biquad_cascade_instance, signal)
    return signal_filtered

def filter_cmsisdsp_q31(signal, numStages, pCoeffs):
    postShift = np.uint8(2)
    pCoeffs_q32 = float_to_fixed_q31(pCoeffs/4)

    # Initialize filter
    biquad_cascade_instance = dsp.arm_biquad_cas_df1_32x64_ins_q31()
    
    status = dsp.arm_biquad_cascade_df1_init_q31(biquad_cascade_instance, numStages, np.array(pCoeffs_q32, dtype=np.int32), np.zeros(numStages*4), postShift)
    # Apply filter
    signal_filtered = dsp.arm_biquad_cascade_df1_q31(biquad_cascade_instance, np.array(signal, dtype=np.int32))
    return signal_filtered


def cmsis_bands_preprocess(audio, samplerate, filter_cmsisdsp, band_ranges, band_order, low_order, q15, downsample_factor=128):
    
    features = []

    if q15:
        audio_16=float_to_fixed_q15(audio)
        audio = fixed_q15_to_fixed_q31(audio_16)

    low_sos = sos_lowpass_filter(lowcut=16, fs=samplerate//downsample_factor, order=low_order) #save for processing in C
    low_numStages, low_pCoeffs = get_cmsis_filter(low_sos)


    for lowcut, highcut in band_ranges:
        # Band filtering
        band_sos = sos_bandpass_filter(lowcut, highcut, fs=samplerate, order=band_order)
        bands_numStages, bands_pCoeffs = get_cmsis_filter(band_sos) #save for processing in C
        
        band_cmsidsp = filter_cmsisdsp(audio, numStages=bands_numStages, pCoeffs=bands_pCoeffs)

        # Getting envelope
        ## Rectify
        rectified_band = np.abs(band_cmsidsp)

        ## Downsampling using Maxpooling with K=stride=downsample_factor
        down_band = [np.max(rectified_band[i:i+downsample_factor]) for i in np.arange(0,rectified_band.shape[0],downsample_factor).astype(int)]
        ## Low-pass filter
        envelope = filter_cmsisdsp(down_band, numStages=low_numStages, pCoeffs=low_pCoeffs)

        normal_envelope = (envelope-np.mean(envelope))/np.std(envelope)
        features.append(normal_envelope)
    return np.array(features).T

def load_features(folder, sample_files, feature, samplerate):
    data = []
    for file_name in sample_files:
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder, file_name)
            audio, sr = librosa.load(file_path, sr=samplerate)
            data_sample = feature(audio, sr)
            data.append(data_sample)
    return np.array(data)