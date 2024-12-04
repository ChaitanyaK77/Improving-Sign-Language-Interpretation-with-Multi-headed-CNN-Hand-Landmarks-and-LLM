from gtts import gTTS
import os

def text_to_speech(text):
    # Initialize gTTS with the text
    tts = gTTS(text=text, lang='en')
    
    # Save the speech as an audio file
    tts.save("output.mp3")
    
    # Play the audio file
    if os.name == 'posix':
        # macOS
        os.system("afplay output.mp3")
    elif os.name == 'nt':
        # Windows
        os.system("start output.mp3")
    else:
        # Linux
        os.system("aplay output.mp3")

# Example usage
text_to_speech("I am Dancing")
