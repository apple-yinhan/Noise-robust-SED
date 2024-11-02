import os
import sys
sys.path.append('/home/yinhan/codes/sep4noiseSED')
import config
from tqdm import tqdm
from LASS_codes.models.clap_encoder import CLAP_Encoder
import torchaudio
import soundfile as sf
from utils import (
    load_ss_model,
    calculate_sdr,
    calculate_sisdr,
    parse_yaml,
    get_mean_sdr_from_dict,
)
import torch
import numpy as np
import io_func as io
import pandas as pd

if __name__ == '__main__':
    print('=== load LASS model ===')
    device = 'cuda:1'
    lass_sr = 16000
    lass_dur = 10
    audio_folder = '/home/yinhan/codes/sep4noiseSED/wildDESED/test10db' 
    # /home/Dataset2/dataset/dcase/2023/task4/DESED/Real/audio/validation/validation_16k/
    # '/home/yinhan/codes/sep4noiseSED/wildDESED/test0db'
    save_folder = '/home/yinhan/codes/sep4noiseSED/wildDESED/test_10db_sep_sedclean' 
    os.makedirs(save_folder, exist_ok=True)
    # '/home/yinhan/codes/sep4noiseSED/demo' # '/home/yinhan/codes/sep4noiseSED/wildDESED/test0db_sep'

    captions = ["alarm bell ringing" , "blender", "cat", "dishes", "dog", "electric shaver toothbrush",
                "frying", "running water", "speech", "vacuum cleaner"]
    # "alarm bell ringing" , "blender", "cat", "dishes", "dog", "electric shaver toothbrush",
    #             "frying", "running water", "speech", "vacuum cleaner"]

    print('LASS sampling rate:', lass_sr)
    if lass_sr == 32000:
        config_yaml = '/home/yinhan/codes/sep4noiseSED/LASS_codes/config/Fsd_Clo_Caps_Autotest_ResUNet_32k.yaml'
        checkpoint_path = '/home/yinhan/codes/sep4noiseSED/LASS_codes/checkpoints/model-epoch=01-val_sdr=8.6049.ckpt'
    elif lass_sr == 16000:
        config_yaml = '/home/yinhan/codes/sep4noiseSED/LASS_codes/config/Fsd_Clo_Caps_Autotest_ResUNet_16k.yaml'
        checkpoint_path = '/home/yinhan/codes/sep4noiseSED/LASS_codes/checkpoints/model-epoch=19-val_sdr=8.1018.ckpt'
    configs = parse_yaml(config_yaml)
    # # Load model
    query_encoder = CLAP_Encoder().eval()

    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_path,
        query_encoder=query_encoder
    ).to(device)

    pl_model.eval()
    with torch.no_grad():
        print('=== separate begin ===')
        est_right_num = 0
        for wavename in tqdm(os.listdir(audio_folder)):
            # wavename = 'Y_AfYlb7oEsg_330.000_340.000.wav'
            # load audio and captions
            audio_path = f'{audio_folder}/{wavename}'
            audio, sr = torchaudio.load(audio_path, channels_first=True)

            if audio.shape[0] == 2:
                audio = (audio[0,:]+audio[1,:])/2
            audio = audio.reshape(1,-1) # [1, samples]

            if sr != lass_sr:
                audio_2 = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=lass_sr)
            else:
                audio_2 = audio

            
            ##  ------ pick captions real according to labels ------
            metadata = pd.read_csv('/home/Dataset2/dataset/dcase/2023/task4/DESED/Real/meta/validation/validation.tsv',
            delimiter='\t')
            # print(metadata.shape) # [nums, 4] filename, onset, offset, event_label
            metadata_pick = metadata[metadata['filename']==wavename]
            # print(metadata_pick['event_label'])
            captions_real_1 = []
            for item in list(metadata_pick['event_label']):
                if isinstance(item, str):
                    if item.replace('_',' ').lower() not in captions_real_1:
                        captions_real_1.append(item.replace('_',' ').lower())
            ## -----------------------------------------------------
            metadata = pd.read_csv('/home/yinhan/codes/sep4noiseSED/weak_predictions/test_-5db_sednoise.txt',
            delimiter='\t', names=['filename', 'event_label'])
            # print(metadata.shape)
            metadata_pick = metadata[metadata['filename']==wavename]
            captions_real_2 = []
            for item in list(metadata_pick['event_label']):
                if isinstance(item, str):
                    if item.lower() not in captions_real_2 and item != 'none':
                        captions_real_2.append(item.lower())
            ## ---------------compare GT and Est -------------------
            est_right = False
            if len(captions_real_1) == len(captions_real_2):
                est_right = True
                for ii in captions_real_2:
                    if ii not in captions_real_1:
                        est_right = False
            if est_right:
                est_right_num += 1
            else:
                pass
                # print(captions_real_1, captions_real_2)
            ## -----------------------------------------------------
            idx = 0
            captions_real = captions_real_2 # captions(all),captions_real_1(GT),captions_real_2(Est) 
            if len(captions_real) != 0:
                for caption in captions_real:
                    conditions = pl_model.query_encoder.get_query_embed(
                                        modality='text',
                                        text=[caption],
                                        device=device 
                                    )

                    
                    segment = audio_2
                    segment = segment.to(device)  # print(segment.shape) [1, 320000]
                    input_dict = {
                                    "mixture": segment[None, :, :],
                                    "condition": conditions,
                                }
            
                    outputs = pl_model.ss_model(input_dict)
                    sep_segment = outputs["waveform"]  # print(sep_segment.shape) [1, 1, 320000]
                    sep_segment = sep_segment[0]
                    # print(torch.mean(sep_segment[0]), torch.std(sep_segment[0]))

                    if idx == 0:
                        final_signal = sep_segment
                    else:
                        final_signal = final_signal + sep_segment
                    idx += 1
            else:
                final_signal = audio_2
                
            if sr != lass_sr:
                final_signal = torchaudio.functional.resample(final_signal, orig_freq=lass_sr, new_freq=sr)
                audio_2 = torchaudio.functional.resample(audio_2, orig_freq=lass_sr, new_freq=sr)

            if len(captions_real) != 0:
                final_signal = final_signal/idx

            sep_savename = f'{wavename[:-4]}'
            io.write_waveform(final_signal, save_folder + f'/{sep_savename}.wav', sr = sr)
            ## io.write_stft_pics(final_signal, save_folder + f'/{sep_savename}.png', sr = sr)
            ## io.write_waveform(audio_2, save_folder + f'/{wavename[:-4]}_ori.wav', sr = sr)
            ## io.write_stft_pics(audio_2, save_folder + f'/{wavename[:-4]}_ori.png', sr = sr)
            ## sys.exit()

        print('Est right ACC: ',est_right_num/1153)
