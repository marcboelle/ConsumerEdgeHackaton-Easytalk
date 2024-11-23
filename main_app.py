from get_audio_function_whisper import get_audio_whisper

import gradio as gr


def translate_audio(audio_path, sex):
    if audio_path is not None:
        new_audio = get_audio_whisper(audio_path, sex)
        return new_audio    
    return new_audio

css_style = """
    #left-audio, #right-audio {
        width: 40%;  /* Set width to 40% of the parent container */
        height:250px;
    }
    #center-button {
        width: 20%;  /* Set the button to take up 20% of the width */
        margin: auto;  /* Center the button */
        height:250px;
    }
    #row-main {
        display: flex;
        /* flex-direction: column;*/
        /justify-content: center; /* Center vertically */
        align-items: center;    /* Optionally center horizontally */
        /*height: 100vh;*/          /* Take full height of the viewport */
    }
""" 

# Interface Gradio

with gr.Blocks(css=css_style) as demo:

    gr.Markdown("<h1 style='text-align: center;'>EasyTalk!</h1>")
   
    gender_selector = gr.Radio(choices=["man", "woman"], label="Select Gender", value="man")
    with gr.Row():
        audio1 = gr.Audio(sources="microphone", label="Input Audio", type="filepath")
        audio2 = gr.Audio(label="Output Audio", type="filepath")
    
    # Associer la fonction à l'événement change sur audio1
    audio1.change(
        fn=translate_audio,
        inputs=[audio1, gender_selector],  # Entrées : audio1 et le genre
        outputs=audio2  # Sortie : audio2
    )

# Lancer l'application
demo.launch()