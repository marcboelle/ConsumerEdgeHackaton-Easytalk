import torch
import librosa
import subprocess
from transformers import MarianMTModel, MarianTokenizer
from IPython.display import Audio, display
import os
import time
from faster_whisper import WhisperModel
from optimum.onnxruntime import ORTModelForSeq2SeqLM

#model = whisper.load_model("base")
model = WhisperModel("tiny", device="cpu", compute_type="int8")
"""
model_name_fr_en = "Helsinki-NLP/opus-mt-fr-en"
helsinki_tokenizer_fr_en = MarianTokenizer.from_pretrained(model_name_fr_en)
#helsinki_model_fr_en = MarianMTModel.from_pretrained(model_name_fr_en)

"""
model_name_fr_en = "Helsinki-NLP/opus-mt-fr-en"
helsinki_tokenizer_fr_en = MarianTokenizer.from_pretrained(model_name_fr_en)
model_name_en_fr = "Helsinki-NLP/opus-mt-en-fr" 
helsinki_tokenizer_en_fr = MarianTokenizer.from_pretrained(model_name_en_fr) 

#helsinki_model_en_fr = MarianMTModel.from_pretrained(model_name_en_fr)

model_name_en_fr = "onnx_quantized_en_to_fr"
helsinki_model_en_fr = ORTModelForSeq2SeqLM.from_pretrained(model_name_en_fr, max_new_tokens = 200)
model_name_fr_en = "onnx_quantized_fr_to_en"
helsinki_model_fr_en = ORTModelForSeq2SeqLM.from_pretrained(model_name_fr_en, max_new_tokens = 200)

# Men models
voice_onnx_man_fr = "fr_FR-tom-medium.onnx"
voice_onnx_config_man_fr = "fr_FR-tom-medium.onnx.json"

voice_onnx_man_en = "en_US-john-medium.onnx"
voice_onnx_config_man_en = "en_US-john-medium.onnx.json"

# Women models
voice_onnx_woman_fr = "fr_FR-upmc-medium.onnx"
voice_onnx_config_woman_fr = "fr_FR-upmc-medium.onnx.json"

voice_onnx_woman_en = "en_US-amy-medium.onnx"
voice_onnx_config_woman_en = "en_US-amy-medium.onnx.json"

# Files
output_file = "output_gr.wav"

# Command to execute Piper
piper_directory = "piper"

def get_audio_whisper(audio_path, sex='man'):

    audio, sample_rate = librosa.load(audio_path, sr=16000)
    """audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # Compute the log-Mel spectrogram
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect language
    _, probs = model.detect_language(mel)

    detected_language = max(probs, key=probs.get)
    print(f"Detected Language: {detected_language}")
    
    result = model.transcribe(audio_path)
    transcription = result["text"]
    """
    segments, info = model.transcribe(audio)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    transcription = " ".join([segment.text for segment in segments])

    if info.language == 'fr':
        language_output = 'en'
    else:
        language_output = 'fr'

    # Load the tokenizer and model

    text = transcription
    print(text)
    
    if language_output == 'fr':
        input_ids = helsinki_tokenizer_en_fr(text, return_tensors="pt").input_ids
        with torch.no_grad():
            translated_ids = helsinki_model_en_fr.generate(input_ids, max_new_tokens = 200)
        # Decode the generated ids to text
        translation = helsinki_tokenizer_en_fr.decode(translated_ids[0], skip_special_tokens=True)
    
    else:
        input_ids = helsinki_tokenizer_fr_en(text, return_tensors="pt").input_ids
        with torch.no_grad():
            translated_ids = helsinki_model_fr_en.generate(input_ids, max_new_tokens=200)
        # Decode the generated ids to text
        translation = helsinki_tokenizer_fr_en.decode(translated_ids[0], skip_special_tokens=True)

    # Define text, model, and output file
    text = translation
    print(text)
    
    if language_output == 'fr':
        if sex == "man":
            voice_onnx, voice_onnx_config = voice_onnx_man_fr, voice_onnx_config_man_fr
        else:
            voice_onnx, voice_onnx_config = voice_onnx_woman_fr, voice_onnx_config_woman_fr
        
    elif language_output == 'en': 
        if sex == "man":
            voice_onnx, voice_onnx_config = voice_onnx_man_en, voice_onnx_config_man_en
        else:
            voice_onnx, voice_onnx_config = voice_onnx_woman_en, voice_onnx_config_woman_en
        
    command = [
        os.path.join(piper_directory, "piper.exe"),  # Le chemin vers piper.exe
        "-m", os.path.join(piper_directory, voice_onnx),  # Modèle à utiliser
        "-c", os.path.join(piper_directory, voice_onnx_config),
        "-f", output_file # Fichier de sortie
    ]

    result = subprocess.run(
        command,  # La commande à exécuter
        input=text,  # Le texte envoyé à Piper
        text=True,  # Pour envoyer le texte en mode chaîne
        check=True  # Lève une exception si la commande échoue
    )

    return output_file

#get_audio_whisper("anas_en.wav")