import torch
import librosa
import subprocess
from transformers import MarianMTModel, MarianTokenizer
from IPython.display import Audio, display
import os
import time
from faster_whisper import WhisperModel
from optimum.onnxruntime import ORTModelForSeq2SeqLM

model = WhisperModel("tiny", device="cpu", compute_type="int8")

model_name_fr_en = "Helsinki-NLP/opus-mt-fr-en"
helsinki_tokenizer_fr_en = MarianTokenizer.from_pretrained(model_name_fr_en)
model_name_en_fr = "Helsinki-NLP/opus-mt-en-fr" 
helsinki_tokenizer_en_fr = MarianTokenizer.from_pretrained(model_name_en_fr) 

model_name_en_fr = "onnx_quantized_en_to_fr"
helsinki_model_en_fr = ORTModelForSeq2SeqLM.from_pretrained(model_name_en_fr, max_new_tokens = 200)
model_name_fr_en = "onnx_quantized_fr_to_en"
helsinki_model_fr_en = ORTModelForSeq2SeqLM.from_pretrained(model_name_fr_en, max_new_tokens = 200)

# Men models
voice_onnx_man_fr = "fr_FR-tom-medium.onnx"
voice_onnx_config_man_fr = "fr_FR-tom-medium.onnx.json"

voice_onnx_man_en = "en_US-john-medium.onnx"
voice_onnx_config_man_en = "en_US-john-medium.onnx.json"


# Command to execute Piper
piper_directory = "piper"

## Tous les anglais to fr
for i in range(10):
    samples_path = "wav_samples"
    filename_en = os.path.join(samples_path, f"recording_english{i}.wav")
    output_file_fr = os.path.join(samples_path, f"output_fr_txt_{i}.txt")
    output_audio_fr = os.path.join(samples_path, f"output_fr_audio_{i}.wav")
    output_file_time_fr = os.path.join(samples_path, f'output_fr_time_{i}.txt')
    
    
    audio, sample_rate = librosa.load(filename_en, sr=16000)

    time_start = time.time()
    
    segments, info = model.transcribe(audio)
    transcription = " ".join([segment.text for segment in segments])
    
    time_finish_transcript = time.time()

    input_ids = helsinki_tokenizer_en_fr(transcription, return_tensors="pt").input_ids
    with torch.no_grad():
        translated_ids = helsinki_model_en_fr.generate(input_ids, max_new_tokens = 200)
    translation = helsinki_tokenizer_en_fr.decode(translated_ids[0], skip_special_tokens=True)
    
    time_finish_translation = time.time()

    command = [
        os.path.join(piper_directory, "piper.exe"),  # Le chemin vers piper.exe
        "-m", os.path.join(piper_directory, voice_onnx_man_fr),  # Modèle à utiliser
        "-c", os.path.join(piper_directory, voice_onnx_config_man_fr),
        "-f", output_audio_fr # Fichier de sortie
    ]

    result = subprocess.run(
        command,  # La commande à exécuter
        input=translation,  # Le texte envoyé à Piper
        text=True,  # Pour envoyer le texte en mode chaîne
        check=True  # Lève une exception si la commande échoue
    )

    time_finish_audio_gen = time.time()

    time_list = [str(time_finish_transcript-time_start), str(time_finish_translation-time_finish_transcript), str(time_finish_audio_gen-time_finish_translation)]
    text_list = [transcription, translation]

    with open(output_file_fr, 'w') as f:
        f.write('\n'.join(text_list))
    
    with open(output_file_time_fr, 'w') as f:
        f.write('\n'.join(time_list))

## Tous les anglais to fr
for i in range(10):
    samples_path = "wav_samples"
    filename_fr = os.path.join(samples_path, f"recording_french{i}.wav")
    output_file_en = os.path.join(samples_path, f"output_en_txt_{i}.txt")
    output_audio_en = os.path.join(samples_path, f"output_en_audio_{i}.wav")
    output_file_time_en = os.path.join(samples_path, f'output_en_time_{i}.txt')

    audio, sample_rate = librosa.load(filename_fr, sr=16000)

    time_start = time.time()
    
    segments, info = model.transcribe(audio)
    transcription = " ".join([segment.text for segment in segments])
    
    time_finish_transcript = time.time()

    input_ids = helsinki_tokenizer_fr_en(transcription, return_tensors="pt").input_ids
    with torch.no_grad():
        translated_ids = helsinki_model_fr_en.generate(input_ids, max_new_tokens = 200)
    translation = helsinki_tokenizer_fr_en.decode(translated_ids[0], skip_special_tokens=True)
    
    time_finish_translation = time.time()

    command = [
        os.path.join(piper_directory, "piper.exe"),  # Le chemin vers piper.exe
        "-m", os.path.join(piper_directory, voice_onnx_man_en),  # Modèle à utiliser
        "-c", os.path.join(piper_directory, voice_onnx_config_man_en),
        "-f", output_audio_en # Fichier de sortie
    ]

    result = subprocess.run(
        command,  # La commande à exécuter
        input=translation,  # Le texte envoyé à Piper
        text=True,  # Pour envoyer le texte en mode chaîne
        check=True  # Lève une exception si la commande échoue
    )

    time_finish_audio_gen = time.time()

    time_list = [str(time_finish_transcript-time_start), str(time_finish_translation-time_finish_transcript), str(time_finish_audio_gen-time_finish_translation)]
    text_list = [transcription, translation]

    with open(output_file_en, 'w') as f:
        f.write('\n'.join(text_list))
    
    with open(output_file_time_en, 'w') as f:
        f.write('\n'.join(time_list))
