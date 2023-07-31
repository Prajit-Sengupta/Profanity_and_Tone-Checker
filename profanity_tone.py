import os
import joblib
from profanity_check import predict_prob 
import matplotlib.pyplot as plt
import streamlit as st 
import flair
from better_profanity import profanity
import joblib
import re

st.set_page_config(layout="wide")
st.write("""# Profanity & Tone Checker""")
st.text('Upload your Podcast Audio File to check for Profanity Words & Tone of the Speaker')

#Upload Audio & Transcript

audio_file= st.file_uploader("Upload Audio File", accept_multiple_files=True)

file_txt =st.file_uploader("Upload Transcript", accept_multiple_files=True)

import librosa
import torch

from transformers import Wav2Vec2ForCTC , Wav2Vec2Tokenizer
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

def transcriptor(audio_file1):
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/sav2vec2-base-960h")

    collection_of_text = []

    # for i in range(1): 
    speech, rate = librosa.load(audio_file1, sr=16000)
    input_values = tokenizer(speech, return_tensors='pt').input_values

    with torch.no.grad():
        logits = model(input_values).logits
    predicted_ids= torch.argaax(logits, dim=1)
    transcription= tokenizer.batch_decode(predicted_ids)[0] 
    collection_of_text.append(transcription)
    final_complete_speech = ""