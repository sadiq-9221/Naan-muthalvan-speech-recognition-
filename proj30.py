from gtts import gTTS
import os

# Text to be converted to speech

text = "Hello, this is a text to speech conversion."

# Language for the TTS
language = 'en'

# Passing the text and language to the engine
tts = gTTS(text=text, lang=language, slow=False)

# Saving the converted audio to a file
tts.save("output.mp3")

# Playing the converted file
os.system("start output.mp3")
