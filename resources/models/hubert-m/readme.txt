This repository contains the best mHuBERT-147 pre-trained model.

MODEL DETAILS: 3rd iteration, K=1000, HuBERT base architecture (95M parameters), 147 languages.

mHuBERT-147 models
mHuBERT-147 are compact and competitive multilingual HuBERT models trained on 90K hours of open-license data in 147 languages. Different from traditional HuBERTs, mHuBERT-147 models are trained using faiss IVF discrete speech units. Training employs a two-level language, data source up-sampling during training. See more information in our paper.

Table of Contents:
Summary
Training Data and Code
ML-SUPERB Scores
Languages and Datasets
Intermediate Checkpoints
Citing and Funding Information
This repository contains:

Fairseq checkpoint (original);
HuggingFace checkpoint (conversion using transformers library);
Faiss index for continuous pre-training (OPQ16_64,IVF1000_HNSW32,PQ16x4fsr).
Related Models:

2nd Iteration mHuBERT-147
1st Iteration mHuBERT-147
CommonVoice Prototype (12 languages)
Training
Manifest list available here. Please note that since training, there were CommonVoice removal requests. This means that some of the listed files are no longer available.

Fairseq fork contains the scripts for training with multilingual batching with two-level up-sampling.

Scripts for pre-processing/faiss clustering available here.

ML-SUPERB Scores
mHubert-147 reaches second and first position in the 10min and 1h leaderboards respectively. We achieve new SOTA scores for three LID tasks. See more information in our paper.

image/png

Languages and Datasets
Datasets: For ASR/ST/TTS datasets, only train set is used.

Aishell and AISHELL-3
BibleTTS
ClovaCall
CommonVoice v11
Google TTS data: Javanese, Khmer, Nepali, Sundanese, South African Languages, Bengali Languages
IISc-MILE: Tamil, Kannada
Japanese Versatile Speech
Kokoro
Kosp2e
Media Speech: Turkish Only
Multilingual LibriSpeech
Samrómur
THCHS-30 and THUYG-20
VoxLingua107
VoxPopuli
Languages present not indexed by Huggingface: Asturian (ast), Basaa (bas), Cebuano (ceb), Central Kurdish/Sorani (ckb), Hakha Chin (cnh), Hawaiian (haw), Upper Sorbian (hsb) Kabyle (kab), Moksha (mdf), Meadow Mari (mhr), Hill Mari (mrj), Erzya (myv), Taiwanese Hokkien (nan-tw), Sursilvan (rm-sursilv), Vallader (rm-vallader), Sakha (sah), Santali (sat), Scots (sco), Saraiki (skr), Tigre (tig), Tok Pisin (tpi), Akwapen Twi (tw-akuapem), Asante Twi (tw-asante), Votic (vot), Waray (war), Cantonese (yue).

Intermediate Checkpoints
For allowing research in training dynamics, the intermediate checkpoints for the three iterations are made available under the CC-BY-NC-SA-4.0 license via a protected link.

Downloading page: https://download.europe.naverlabs.com/mhubert147/
User: user
Password: license mentioned above in bold
Citing and Funding Information
@inproceedings{boito2024mhubert,
author={Boito, Marcely Zanon and Iyer, Vivek and Lagos, Nikolaos and Besacier, Laurent and Calapodescu, Ioan},
title={{mHuBERT-147: A Compact Multilingual HuBERT Model}},
year=2024,
booktitle={Interspeech 2024},

This is an output of the European Project UTTER (Unified Transcription and Translation for Extended Reality) funded by European Union’s Horizon Europe Research and Innovation programme under grant agreement number 101070631.
For more information please visit https://he-utter.eu/

NAVER LABS Europe: https://europe.naverlabs.com/