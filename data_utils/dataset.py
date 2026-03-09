import h5py
import yaml
import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.utils import shuffle

main_dir="Data/Splits/"

def clip_audio(samples, sr, target_duration):
    # Taken from Ilaria's code
    if samples.ndim > 1:
        samples = samples[:, 0]  # Convert to mono if stereo
    
    target_length = int(target_duration * sr)
    if len(samples) < target_length:
        samples = librosa.util.pad_center(samples, size=target_length, axis=0)
    elif len(samples)< 2.5*sr: #TODO fix this
        samples = samples[:target_length]
    else:
        samples = samples[int(0.5*sr):int(2*sr)] #TODO fix this
    return samples



class YellowhammerData:
    def __init__(self, split_name, key, duration, augment, transform, sample_rate=20480):
        self.split_name = split_name
        self.sample_rate = sample_rate
        self.key = key
        self.duration = duration
        self.transform = transform
        self.augment = augment
        self.split_path = os.path.join(main_dir, self.split_name)

        self.data_path = os.path.join(self.split_path, key+"/")
        self.label_file = os.path.join(self.data_path, "labels.csv")

        if not os.path.exists(self.data_path):
            print("Preparing dataset ...")
            os.makedirs(self.data_path)
            self.get_data(augment)
        else:
            print("Data will be read from "+self.data_path)
        
        audio_files = os.listdir(self.data_path)
        self.audio_files= [f for f in audio_files if ".wav" in f]
        
        if transform is not None:
            self.transform_path=os.path.join(self.data_path, transform.name)
            if not os.path.exists(self.transform_path):
                print("Preparing "+transform.name+" ...")
                os.makedirs(self.transform_path)
                self.get_transform()

            else:
                print(transform.name+" will be loaded from "+self.transform_path)
    
    def get_label(self, l, binary):
        songs = ['A', 'D', 'E', 'F', 'G', 'H']
        if 'no_YH' in l:
            return 0
        elif binary:
            return 1
        else:
            check = [s in l for s in songs]
            return np.where(check)[0][0]+1


    def parse_distance(self,file_name):
        # Get distance as a float
        distance_strings = ["_6_5m_", "_12_5m", "_25m_", "_50m_", "_100m_", "_150m_", "_200m_"]
        distance_values = np.array([6.5, 12.5, 25, 50, 100, 150, 200])

        check_distance = np.array([d in file_name for d in distance_strings])

        distance = distance_values[check_distance]
        if "YH" in file_name and distance.shape[0] != 0:
            return distance[0]
        elif "bioneg_" in file_name:
            return -1
        else:
            return -2
        
    def load_audio(self, binary, with_dist=False, norm=True):
        audio_X = []
        for audio_file in self.audio_files:
            audio_path = os.path.join(self.data_path, audio_file)
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            if norm:
                audio = (audio - np.mean(audio))/np.std(audio)
            audio_X.append(audio)
        audio_X=np.array(audio_X)

        #TODO fix this section 
        df_labels=pd.read_csv(self.label_file)
        label_map = dict(zip(df_labels["fnames"], df_labels["labels"]))
        save_names=[name.split(".")[-2] for name in self.audio_files]
        labels = [self.get_label(label_map[f], binary=binary) for f in save_names] 

        if with_dist:
            distances = np.array([self.parse_distance(name) for name in save_names])
            return shuffle(audio_X[...,np.newaxis], np.array(labels), distances)
        else:
            return shuffle(audio_X[...,np.newaxis], np.array(labels))

    def load_feature(self, binary, with_dist=False):
        feature_files = os.listdir(self.transform_path)
        feature_files= [f for f in feature_files if ".npy" in f]

        feature_X=[]
        for feature_fname in feature_files:
            feature_path = os.path.join(self.transform_path, feature_fname)
            feature = np.load(feature_path)
            feature_X.append(feature)
        
        feature_X=np.array(feature_X)

        #TODO fix this section 
        df_labels=pd.read_csv(self.label_file)
        label_map = dict(zip(df_labels["fnames"], df_labels["labels"]))
        save_names=[name.split(".")[-2] for name in feature_files]
        labels = [self.get_label(label_map[f], binary=binary) for f in save_names] 
    
        if with_dist:
            distances = np.array([self.parse_distance(name) for name in save_names])
            return shuffle(feature_X[...,np.newaxis], np.array(labels), distances)

        else:
            return shuffle(feature_X[...,np.newaxis], np.array(labels))
    
    def get_transform(self):
        for audio_file in self.audio_files:
            audio_path = os.path.join(self.data_path, audio_file)
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            feature = self.transform.feature(audio)
            feature_name=audio_file.split(".")[-2]
            np.save(os.path.join(self.transform_path, feature_name), feature)
        
    def get_data(self, augment):
        sources_yaml = os.path.join(self.split_path, self.split_name+".yaml")
        
        with open(sources_yaml, 'r') as f:
            sources = yaml.load(f, Loader=yaml.SafeLoader)        
        meta_df = pd.read_csv(os.path.join(self.split_path, self.key+"_data.csv"))
        paths_to_samples = meta_df.apply(lambda r: os.path.join(sources[r["source"]], r["file"]), axis=1)
        labels=meta_df['song'].to_numpy()

        new_names=[]
        new_labels=[]
        for path_og, label in zip(paths_to_samples, labels):

            f_name= path_og.split("/")[-1]
            sample, sr = librosa.load(path_og, sr=self.sample_rate)
            sample = clip_audio(sample, sr=sr, target_duration=self.duration)
            new_labels.append(label)
            new_names.append(f_name)

            sf.write(os.path.join(self.data_path, f_name), sample, samplerate=sr)

            if augment is not None and label!="no_YH" and self.key=='train':
                augmented_x, aug_names = self.augment(f_name, sample, sr=self.sample_rate)
                for aug_audio, aug_name in zip(augmented_x, aug_names):
                    sf.write(os.path.join(self.data_path,aug_name), aug_audio, sr)
                
                new_labels.extend([label for n in aug_names])
                new_names.extend(aug_names)
        
        save_names=[name.split(".")[-2] for name in new_names]
        pd.DataFrame({'fnames': save_names, 'labels': new_labels}).to_csv(self.label_file, index=False)




class YellowhammerData_old:
    def __init__(self, split_name, key, duration, augment, transform, sample_rate=20480):
        self.split_name = split_name
        self.key = key
        self.sample_rate = sample_rate
        self.duration = duration
        self.transform = transform
        self.augment = augment
        self.split_path = os.path.join(main_dir, self.split_name)
        self._datapath = os.path.join(self.split_path, key+"_data.h5f")

        self.get_data(augment=(augment is not None))
        if transform is not None:
            self.get_features()
        
    def get_split(self, augment,save):
        sources_yaml = os.path.join(self.split_path, self.split_name+".yaml")
        
        with open(sources_yaml, 'r') as f:
            sources = yaml.load(f, Loader=yaml.SafeLoader)        
        meta_df = pd.read_csv(os.path.join(self.split_path, self.key+"_data.csv"))
        paths_to_samples = meta_df.apply(lambda r: os.path.join(sources[r["source"]], r["file"]), axis=1)
        

    def get_data(self, augment=True, save=True):
        if os.path.isfile(self._datapath):
            self.load_split()
        else:
            self.get_split(augment=augment, save=save)
        return self.X[...,np.newaxis], self.y


    def get_label(self, l):
        songs = ['A', 'D', 'E', 'F', 'G', 'H']
        if 'no_YH' in l:
            return 0
        else:
            check = [s in l for s in songs]
            return np.where(check)[0][0]+1
        
    def get_distances(self):
        meta_df = pd.read_csv(os.path.join(self.split_path, self.key+"_data.csv"))
        return meta_df["dist"].to_numpy()


    def get_split(self, augment,save):
        sources_yaml = os.path.join(self.split_path, self.split_name+".yaml")
        
        with open(sources_yaml, 'r') as f:
            sources = yaml.load(f, Loader=yaml.SafeLoader)        
        meta_df = pd.read_csv(os.path.join(self.split_path, self.key+"_data.csv"))
        paths_to_samples = meta_df.apply(lambda r: os.path.join(sources[r["source"]], r["file"]), axis=1)
        
        self.X, self.y = [],[]
        i=0
        labels=meta_df['song'].to_numpy() #TODO fix this
        for f in paths_to_samples:
            label=labels[i]
            i=i+1

            sample, sr = librosa.load(f, sr=self.sample_rate)
            sample = clip_audio(sample, sr=self.sample_rate, target_duration=self.duration)
            if augment and label!="no_YH":
                augmented_x = self.augment(sample, sr=self.sample_rate)
                self.X.extend(augmented_x)
                self.y.extend([label for x in augmented_x])
            
            self.X.append(sample)
            self.y.append(label)
        

        self.X = np.array(self.X)
        self.y = [self.get_label(l) for l in self.y]
        self.y = np.array(self.y).astype(int)
        self.X, self.y = shuffle(self.X, self.y )
        if save:
            h5_file = h5py.File(self._datapath, 'w')
            h5_file.create_dataset('data', data=self.X)
            h5_file.create_dataset('labels', data=self.y)
            h5_file.close()
        
    def load_split(self):
        h5_file = h5py.File(self._datapath, 'r')
        X_data, labels = np.array(h5_file['data']), np.array(h5_file['labels'])
        self.X = np.array(X_data)
        self.y = np.array(labels)
    
    def get_features(self, save=True):
        features_path = os.path.join(self.split_path, self.key+"_mel_data.h5f")
        if os.path.isfile(features_path):
            h5_file = h5py.File(features_path, 'r')
            X_features = np.array(h5_file['data'])
        else:     
            X_features = np.array([self.transform(s) for s in self.X])[...,np.newaxis]
            if save:
                h5_file = h5py.File(features_path, 'w')
                h5_file.create_dataset('data', data=X_features)
                h5_file.create_dataset('labels', data=self.y)
                h5_file.close()
        return X_features
    
class PytorchYellowhammerData():
    def __init__(self, split_name, key, sample_rate=20480, duration=1.5, transform=None):
        
        self.sample_rate = sample_rate
        self.duration = duration
        self.transform = transform

        split_path=os.path.join(main_dir, split_name)
        sources_yaml = os.path.join(split_path, split_name+".yaml")
        
        with open(sources_yaml, 'r') as f:
            sources = yaml.load(f, Loader=yaml.SafeLoader)        
        meta_df = pd.read_csv(os.path.join(split_path, key+"_data.csv"))
        self.paths_to_samples = meta_df.apply(lambda r: os.path.join(sources[r["source"]], r["file"]), axis=1)