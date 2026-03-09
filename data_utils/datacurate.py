# Code for creating the different splits
import os
import re
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

def meta_transmission(folder):
    def get_label(f):
        song = ['A', 'D', 'E', 'F', 'G', 'H']
        init_phase = ['_'+p+"_" for p in song]

        if 'bioneg' in f:
            return -1
        elif 'xmin' in f:
            return -2
        else:
            check = [s in f for s in init_phase]
            return song[np.where(check)[0][0]]

    def parse_distance(file_name):
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
    
    def parse_indiv(file_name): 
        # Get indivual ID number
        sub_strings = np.array(file_name.split("_"))
        index = np.where(sub_strings == 'YH')[0][0]
        return int(sub_strings[index+1])

    file_names = os.listdir(folder)
    file_names = [f for f in file_names if ".wav" in f]
    song_labels = [get_label(f) for f in file_names]
    indivs = [parse_indiv(f) for f in file_names]
    distances = [parse_distance(f) for f in file_names]
    return pd.DataFrame({
        'file': file_names,
        'song': song_labels,
        'Indiv': indivs,
        'dist': distances,
    })

def meta_ID_Set(folder):
    parent_folders = os.listdir(folder)
    parent_folders = [f for f in parent_folders if ".DS_Store" not in f]

    file_names, indivs, song_labels = [], [], []

    for parent in parent_folders:
        match = re.match(r"(\d+)(.*)", parent)
        assert(match)
        indiv = int(match.group(1))
        song = match.group(2)
        samples = os.listdir(os.path.join(folder,parent))
        samples = [os.path.join(parent, s) for s in samples if ".wav" in s]
        file_names.extend(samples)
        indivs.extend([indiv for s in samples])
        song_labels.extend([song for s in samples])

    augment = [False for f in file_names]
    return pd.DataFrame({
        'file': file_names,
        'song': song_labels,
        'Indiv': indivs,
        'augment': augment,
    })

def meta_nontarget(folder):
    #TODO expand this when we add more challenging negatives
    file_names = os.listdir(folder)
    file_names = [f for f in file_names if ".wav" in f]
    isBio = ["bioneg_" in f for f in file_names]
    song_labels = ["no_YH" for f in file_names]
    return pd.DataFrame({
        'file': file_names,
        'song': song_labels,
        'isBio': isBio,
    })

def data_split(main_source, test_source=None, nontarget_source=None):
    
    train_df = None
    test_df = None

    if nontarget_source is not None:
        non_folder = nontarget_source['folder']
        df_non = nontarget_source['meta'](non_folder)
        df_non.insert(1, "source", nontarget_source['source'])
        neg_train, neg_test = train_test_split(
            df_non,
            test_size=0.2,
            shuffle=True
        )
        train_df = neg_train if train_df is None else pd.concat([train_df, neg_train], ignore_index=True)
        test_df  = neg_test if test_df  is None else pd.concat([test_df,  neg_test], ignore_index=True)
        
    if test_source is None:
        main_folder = main_source['folder']
        df_main = main_source['meta'](main_folder)
        df_main.insert(1, "source", main_source['source'])
        tr_df, te_df = train_test_split(
            df_main,
            test_size=0.2,
            shuffle=True
        )

        train_df = tr_df if train_df is None else pd.concat([train_df, tr_df], ignore_index=True)
        test_df  = te_df if test_df  is None else pd.concat([test_df,  te_df], ignore_index=True)

    else:
        main_folder = main_source['folder']
        tr_df = main_source['meta'](main_folder)
        tr_df.insert(1, "source", main_source['source'])

        test_folder = test_source['folder']
        te_df = test_source['meta'](test_folder)
        te_df.insert(1, "source", test_source['source'])

        train_df = tr_df if train_df is None else pd.concat([train_df, tr_df], ignore_index=True)
        test_df  = te_df if test_df  is None else pd.concat([test_df,  te_df], ignore_index=True)
    
    return train_df, test_df


def data_curate(splits_folder, split_name, main_source, test_source=None, nontarget_source=None):
    split_path=os.path.join(splits_folder, split_name)
    os.makedirs(split_path)

    train_df, test_df = data_split(main_source, test_source=test_source, nontarget_source=nontarget_source)
    train_df.to_csv(os.path.join(split_path, "train_data.csv"), index=False)
    test_df.to_csv(os.path.join(split_path, "test_data.csv"), index=False)

    split = {
        main_source["source"]: main_source["folder"]
    }

    if test_source is not None:
        split[test_source["source"]] = test_source["folder"]

    if nontarget_source is not None:
        split[nontarget_source["source"]] = nontarget_source["folder"]

    yaml_path = os.path.join(split_path, split_name+'.yaml')
    with open(yaml_path, 'w') as file:
        documents = yaml.dump(split, file, default_flow_style=False)



if __name__ == "__main__":
    splits_save = "Data/Splits/"
    transmission_test ={
        'source': 'transmission_test',
        'folder':"Data/Transmission_test/positive/",
        'meta':meta_transmission
    }
    training_ilaria ={
            'source': 'training_ilaria',
            'folder':"Data/Yell_ID_Set/Training_Set/",
            'meta':meta_ID_Set
        }
    negatives ={
            'source': 'negatives',
            'folder':"Data/Transmission_test/negative/",
            'meta':meta_nontarget
        }

    data_curate(splits_save, "full_transmission_split", main_source=transmission_test, test_source=None, nontarget_source=negatives)
    data_curate(splits_save, "transmission_split", main_source=transmission_test, test_source=None, nontarget_source=None)
    data_curate(splits_save, "full_ID_Set_split", main_source=training_ilaria, test_source=transmission_test, nontarget_source=negatives)
    data_curate(splits_save, "ID_Set_split", main_source=training_ilaria, test_source=transmission_test, nontarget_source=None)