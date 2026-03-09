import os
import librosa
import textgrid
import numpy as np
import soundfile as sf

def audio_clips(folder, outfolder, suffix="", maxclips=-1, dir_exist_ok=False, augment=False, clipsize=0.5, sr=None):
    # Divide recordings into clipsize(seconds) clips

    os.makedirs(outfolder, exist_ok=dir_exist_ok)
    
    files = os.listdir(folder)
    files = [f for f in files if ".wav" in f]
    files = [f for f in files if "speaker" not in f]

    for sound_file in files:
        data, samplerate = librosa.load(folder+sound_file, sr=sr)

        if data.shape[0]/samplerate >= clipsize:
            nb_clips = int(data.shape[0]/(samplerate*clipsize))
            if maxclips > 0:
                nb_clips = np.min([nb_clips,maxclips])

            start = 0
            end = int(clipsize*samplerate)
            for clip in range(nb_clips):
                data_clipped = data[start:end]
                clip_filename=suffix+"clip_"+str(clip+1)+"_"+sound_file

                sf.write(outfolder+clip_filename, data_clipped, samplerate)
                
                start=end
                end = end + int(clipsize*samplerate)

def prepare_negatives(grid_folder, negative_folder, clipsize):
    # Extract negatives from the full recordings
    if not os.path.exists(negative_folder):
        os.makedirs(negative_folder)

        files = os.listdir(grid_folder)
        files = [f for f in files if ".wav" in f]
        files = [f for f in files if "speaker" not in f]
        
        for sound_file in files:
            data, samplerate = librosa.load(grid_folder+sound_file, sr=None)
            
            grid_file = grid_folder + sound_file[:-4]+".TextGrid"
            tg = textgrid.TextGrid.fromFile(grid_file)

            k = 0
            for i in range(len(tg.tiers)):
                for j in range(len(tg.tiers[i])):
                    if "YH" not in tg.tiers[i][j].mark:
                        negative = 'none'
                        if len(tg.tiers[i][j].mark.strip())!=0:
                            negative = tg.tiers[i][j].mark
                        xmin = tg.tiers[i][j].minTime
                        xmax = tg.tiers[i][j].maxTime
                        
                        data_negative = data[int(xmin*samplerate):int(xmax*samplerate)]
                        if data_negative.shape[0]/samplerate >= clipsize:
                            nb_clips = int(data_negative.shape[0]/(samplerate*clipsize))

                            start = 0
                            end = int(clipsize*samplerate)
                            for clip in range(nb_clips):
                                data_clipped = data_negative[start:end]
                                sf.write(negative_folder+"xmin_"+str(round(xmin, 2))+"_"+"xmax_"+str(round(xmax, 2))+"_"+"clip_"+str(clip+1)+"_"+negative+"_"+str(k)+"_"+sound_file, data_clipped, samplerate)
                                start=end
                                end = end + int(clipsize*samplerate)
                            k=k+1

    else:
        print("Folder: "+negative_folder+" exists already.")

if __name__ == "__main__":

    negative_folder="Data/Transmission_test/negative/"
    positive_folder="Data/Transmission_test/positive/"

    songs_folder="Data/Training_set_YH_songs/ALL_Songs_CUT/"
    grid_folder = "Data/Training_set_YH_songs/recordings_textgrid_on_axis/"
    bioneg_folder = "Data/Training_set_YH_songs/Negative_samples_241016/"

    CLIP_SIZE=1.5

    if not os.path.isdir(positive_folder):
        audio_clips(songs_folder, positive_folder, suffix="", clipsize=CLIP_SIZE, maxclips=-1, dir_exist_ok=False)

    if not os.path.isdir(negative_folder):
        prepare_negatives(grid_folder, negative_folder, clipsize=CLIP_SIZE)
        audio_clips(bioneg_folder, negative_folder, suffix="bioneg_", clipsize=CLIP_SIZE, maxclips=-1, dir_exist_ok=True)