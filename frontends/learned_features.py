import numpy as np
import cmsisdsp as dsp
import tensorflow as tf

class CMSISConv1D:
    
    def __init__(self, stride, npy_int16_path, model_path=None, layer_idx=None):
        
        if model_path is not None:
            raw_cnn_block=tf.keras.models.load_model(model_path)
            conv_layer=raw_cnn_block.get_weights()[layer_idx]
            conv_layer = np.squeeze(conv_layer).T
            self.kernels=dsp.arm_float_to_q15(conv_layer).reshape(conv_layer.shape)
        else:
            conv_layer=np.load(npy_int16_path)
            conv_layer = np.squeeze(conv_layer)
            self.kernels=conv_layer
            idx_start, idx_end = self.get_prune_idx()
            self.kernels=conv_layer[:, idx_start:idx_end]
            
        self.start_idx=conv_layer.shape[1] - self.kernels.shape[1]


        self.nb_filters=self.kernels.shape[0]
        self.kernel_len=self.kernels.shape[1]
        self.stride=stride
        
    def get_prune_idx(self):
        prune_start, prune_end=[],[]
        for kernel in self.kernels:
            non_zero_line=np.where(kernel!=0)
            prune_start.append(np.min(non_zero_line))
            prune_end.append(np.max(non_zero_line))

        return np.min(prune_start), np.max(prune_end)

    def get_padding(self, L):
        L_out=int(L/self.stride)
        P=max((L_out - 1) * self.stride + self.kernel_len - L, 0)
        pad_left = P // 2
        pad_right = P - pad_left
        return pad_left, pad_right 
    
    def get_out_w(self, L):
        return int((L - self.kernel_len)/self.stride) + 1
    
    def feature(self, audio_input):
        
        audio_int16 = dsp.arm_float_to_q15(audio_input)

        L=audio_input.shape[0]
        pad_left, pad_right=self.get_padding(L)
        
        audio_padded = np.pad(audio_int16, (pad_left, pad_right), mode='constant')
        
        out_w= self.get_out_w(len(audio_padded))
        results=np.zeros((self.nb_filters, out_w))

        for idx_out in range(out_w):
            start_idx=idx_out*self.stride
            input_clip = audio_padded[start_idx:start_idx+self.kernel_len]

            for idx_k in range(self.nb_filters):
                results[idx_k, idx_out] = dsp.arm_dot_prod_q15(self.kernels[idx_k], input_clip)
        
        return results.reshape((self.nb_filters, out_w))