from pydub import AudioSegment
import subprocess

import os 

# Load the .m4a file
folder_path = "m4a_samples"
output_path = "wav_samples"
if not os.path.exists(output_path):
    os.makedirs(output_path)

list_audio = [os.path.join(folder_path, f"Recording ({i}).m4a") for i in range(9,19)]

for i, audio in enumerate(list_audio):
    subprocess.run(["ffmpeg", "-i", audio, os.path.join(output_path, f"recording_english{i}.wav")], check=True)