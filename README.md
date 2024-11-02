# Leveraging LLM and Text-Queried Separation for Noise-Robust Sound Event Detection
This is the official code for paper **"Leveraging LLM and Text-Queried Separation for Noise-Robust Sound Event Detection"**.

Paper Link: 

- Please first `pip install -r requirements.txt`

## 1. Pre-trained LASS model
Please follow instructions in DCASE 2024 Challenge Task 9 Baseline for pre-training LASS models: [https://github.com/Audio-AGI/dcase2024_task9_baseline](https://github.com/Audio-AGI/dcase2024_task9_baseline)
                                                                                                                                                        
                                                                                                                                                        
                                                                                                                                                         
 Or **Download our pre-trained AudioSep-Dp model**, we release our pre-trained AudioSep-DP model at:                                                                                             
                                                                                                                                                        
## 2. Sound Event Detection

### 2.1 Prepare data 
                                                                       
In our paper, we used **DESED**, **AudioSet-Strong**, and **WildDESED** datasets. Please download dataset from [https://project.inria.fr/desed/](https://project.inria.fr/desed/), [https://research.google.com/audioset/](https://research.google.com/audioset/), and [https://zenodo.org/records/14013803](https://zenodo.org/records/14013803)

### 2.2 Use pre-trained LASS models for separation

In our work, we used the pre-trained LASS model to extract sound tracks of different events. Please use `separate_audio.py` to perform this procedure.
                                                                                                  
### 2.3 Training and Evaluation

**Without curriculum learning**: run `python train_pretrained.py`

**With curriculum learning**: run `python train_pretrained_cl.py`

**PS** Please follow instructions in [DCASE 2024 Task 4](https://dcase.community/challenge2024/task-sound-event-detection-with-heterogeneous-training-dataset-and-potentially-missing-labels) to extract embeddings through pre-trained model **BEATs**.
                                                                                                  
