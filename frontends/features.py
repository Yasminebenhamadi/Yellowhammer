import librosa
import numpy as np
import cmsisdsp as dsp
from scipy.signal import butter, windows
from numpy.lib.stride_tricks import sliding_window_view
LOG_EPS=1e-10

# TODO for ESP

# For devices running CMSIS-DSP 

class MelSpecCMSIS:
    def __init__(self, samplerate, window_len, window_stride, nb_mels, fmin, fmax):
        self.name = f"MelSpecCMSIS_sr{samplerate}_win{window_len}_hop{window_stride}_mels{nb_mels}_fmin{fmin}_fmax{fmax}"
        self.samplerate = samplerate
        self.window_len = window_len
        self.window_stride = window_stride
        self.nb_mels = nb_mels
        self.fmin = fmin
        self.fmax = fmax

        self.fft_istance = dsp.arm_rfft_fast_instance_f32()
        init_STATUS = dsp.arm_rfft_fast_init_f32(self.fft_istance, window_len)
        assert(not init_STATUS)

        self.hann_window = windows.hann(window_len)
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
        
        return dsp.arm_vlog_f32(mel_spec + LOG_EPS)

    def feature(self, audio, standard=True):
        v = sliding_window_view(audio, self.window_len)[::self.window_stride]
        outs = []
        for row in v:
            outs.append(self._process_window(row))
        
        if standard:
            return np.array(outs).T - 2*np.log(np.std(audio))
        else:
            return np.array(outs).T


class BandEnvelopeCMSIS:
    def __init__(self, samplerate, band_ranges, q31, band_order=2, low_order=6, downsample_factor=128, lowcut=16):
        self.name="BandEnvelopeCMSIS"
        self.samplerate=samplerate
        self.band_ranges=band_ranges
        self.band_order=band_order
        self.low_order=low_order
        self.q31=q31 
        self.downsample_factor=downsample_factor
        self.lowcut=lowcut

        if q31:
            self.postShift = np.uint8(2)

        #Get DSP-CMSIS filters
        low_sos = self.sos_lowpass_filter(lowcut=lowcut, fs=self.samplerate//self.downsample_factor, order=self.low_order) #save for processing in C
        self.low_numStages, self.low_pCoeffs = self.get_cmsis_filter(low_sos)

        if q31:
            self.low_pCoeffs = dsp.arm_float_to_q31(self.low_pCoeffs/(1<<self.postShift))

        self.band_filters_cmsis = []
        
        for lowcut, highcut in self.band_ranges:
            band_sos = self.sos_bandpass_filter(lowcut, highcut, fs=self.samplerate, order=self.band_order)
            bands_numStages, bands_pCoeffs = self.get_cmsis_filter(band_sos)
            
            if q31:
                bands_pCoeffs = dsp.arm_float_to_q31(bands_pCoeffs/(1<<self.postShift))

            self.band_filters_cmsis.append((bands_numStages, bands_pCoeffs))
    
    @staticmethod
    def sos_bandpass_filter(lowcut, highcut, fs, order):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut /  nyquist
        sos = butter(order, [low, high], btype = 'band', output = 'sos')
        return sos
    
    @staticmethod
    def sos_lowpass_filter(lowcut, fs, order=6):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        sos = butter(order, low, btype = 'low', analog=False, output = 'sos')
        return sos
    
    @staticmethod
    def get_cmsis_filter(sos):
        numStages = sos.shape[0]
        new_shape = numStages*(sos.shape[1] - 1)
        pCoeffs=np.reshape(np.hstack((sos[:,:3],-sos[:,4:])), new_shape)
        return numStages, pCoeffs

    def filter_cmsisdsp_f32(self,signal, numStages, pCoeffs):
        # Initialize filter
        biquad_cascade_instance = dsp.arm_biquad_cascade_df2T_instance_f32()
        status = dsp.arm_biquad_cascade_df2T_init_f32(biquad_cascade_instance, numStages, pCoeffs, np.zeros(numStages*4))
        assert(not status)
        # Apply filter
        signal_filtered = dsp.arm_biquad_cascade_df2T_f32(biquad_cascade_instance, signal)
        return signal_filtered

    def filter_cmsisdsp_q31(self,signal, numStages, pCoeffs):
        # Initialize filter
        biquad_cascade_instance = dsp.arm_biquad_cas_df1_32x64_ins_q31()
        
        status = dsp.arm_biquad_cascade_df1_init_q31(biquad_cascade_instance, numStages, np.array(pCoeffs, dtype=np.int32), np.zeros(numStages*4), self.postShift)
        assert(not status)
        
        # Apply filter
        signal_filtered = dsp.arm_biquad_cascade_df1_q31(biquad_cascade_instance, np.array(signal, dtype=np.int32))
        return signal_filtered

    def feature(self, audio):

        filter_cmsisdsp = self.filter_cmsisdsp_q31 if self.q31 else self.filter_cmsisdsp_f32
        
        features = []

        if self.q31:
            audio_16=dsp.arm_float_to_q15(audio)  # simply because audio is fed in int16 (mostly unnecessary)
            audio = dsp.arm_q15_to_q31(audio_16)

        for bands_numStages, bands_pCoeffs in self.band_filters_cmsis:
            band_cmsidsp = filter_cmsisdsp(audio, numStages=bands_numStages, pCoeffs=bands_pCoeffs)

            # Getting envelope

            ## Rectify
            rectified_band = np.abs(band_cmsidsp)

            ## Downsampling using Maxpooling with K=stride=downsample_factor
            down_band = [np.max(rectified_band[i:i+self.downsample_factor]) for i in np.arange(0,rectified_band.shape[0],self.downsample_factor).astype(int)]
            
            ## Low-pass filter
            envelope = filter_cmsisdsp(down_band, numStages=self.low_numStages, pCoeffs=self.low_pCoeffs)

            normal_envelope = (envelope-np.mean(envelope))/np.std(envelope)
            features.append(normal_envelope)
        return np.squeeze(np.array(features).T)