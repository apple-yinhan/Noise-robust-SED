import soundfile as sf
import moviepy.editor as mp
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import json

def write_waveform(waveform, save_path, sr = 44100):
    # waveform: [channel, samples]
    waveform = waveform.T
    waveform = waveform.detach().cpu().numpy()
    sf.write(save_path, waveform, sr, 'PCM_16')

def extract_audio_from_video(video_path, save_folder):
    filename = os.path.basename(video_path)
    filename_without_ext = filename.rsplit('.', 1)[0]
    new_name = filename_without_ext + ".wav"
    my_clip = mp.VideoFileClip(video_path)
    my_clip.audio.write_audiofile(save_folder + f'/{new_name}')

def write_stft_pics(waveform, save_path, sr = 44100):
    if waveform.shape[0] == 1:
        waveform = waveform.reshape((-1,)).detach().cpu().numpy()
        stft = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max)

        # === analyze process ===
        # print(stft.shape) # [1024, frames]
        # print(np.min(stft), np.max(stft)) # -80, -3e-6
        # min_value = np.min(stft)
        # a = np.sum(stft<=(min_value+5))
        # b = stft.shape[0] * stft.shape[1]
        # print(a / b) 
        # =======================

        plt.figure()
        librosa.display.specshow(stft, y_axis='linear', x_axis='time', sr=sr)
        # plt.colorbar()
        plt.tight_layout()
        plt.savefig(save_path, dpi=600)
    else:
        waveform_0 = waveform[0].reshape((-1,)).detach().cpu().numpy()
        stft_0 = librosa.amplitude_to_db(np.abs(librosa.stft(waveform_0)), ref=np.max)

        waveform_1 = waveform[1].reshape((-1,)).detach().cpu().numpy()
        stft_1 = librosa.amplitude_to_db(np.abs(librosa.stft(waveform_1)), ref=np.max)
        plt.figure()
        plt.subplot(2,1,1)
        librosa.display.specshow(stft_0, y_axis='linear', x_axis='time', sr=sr)
        # plt.colorbar()
        plt.subplot(2,1,2)
        librosa.display.specshow(stft_1, y_axis='linear', x_axis='time', sr=sr)
        # plt.colorbar()
        plt.tight_layout()
        plt.savefig(save_path, dpi=600)

def save_list_to_json(lst, file_path):
    with open(file_path, 'w') as file:
        json.dump(lst, file)

def load_json_file(json_path):
    with open(json_path, 'r') as fcc_file:
        fcc_data = json.load(fcc_file)
    return fcc_data