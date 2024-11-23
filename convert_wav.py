from pydub import AudioSegment

# Chemin vers le fichier .m4a et le fichier de sortie .wav
input_file = "recording.m4a"  # Remplacez par votre fichier
output_file = "input"

try:
    # Charger le fichier .m4a
    audio = AudioSegment.from_file(input_file, format="m4a")
    
    # Exporter en .wav
    audio.export(output_file, format="wav")
    print(f"Conversion réussie ! Fichier .wav sauvegardé dans : {output_file}")
except Exception as e:
    print(f"Erreur lors de la conversion : {e}")