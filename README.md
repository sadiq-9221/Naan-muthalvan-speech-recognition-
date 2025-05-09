# Naan-muthalvan-speech-recognition-
1. Building an End-to-End Speech Recognition Pipeline: Signal Processing, Acoustic Modeling, and Performance Evaluation (FEATURES EXTRACTION)
Project Description:
This project implements a foundational speech recognition pipeline focusing on feature extraction techniques using audio recordings. The system processes audio files to extract Mel-Frequency Cepstral Coefficients (MFCCs), which are essential for training acoustic models such as HMMs or deep learning models.

Key Components:

Preprocessing of audio signals (resampling, normalization)

Extraction of MFCC features

Dataset: Spoken digit dataset (e.g., FSDD or similar)

Output: NumPy arrays of MFCC features ready for training

Technologies Used:

Python

Librosa

NumPy

Matplotlib

Usage:

Place dataset in the specified folder.

Run feature_extraction.py.

Extracted features are saved for use in downstream tasks (e.g., model training).

2. Real-Time Speech-to-Text System for Customer Support Automation (TEXT TO SPEECH - TTS)
Project Description:
A real-time TTS (Text-to-Speech) system tailored for customer support automation. This converts dynamically generated or predefined text responses into speech, enabling responsive voice agents.

Key Components:

Text input interface (could be CLI or backend-fed)

Text-to-Speech conversion

Real-time audio playback

Technologies Used:

Python

pyttsx3 (offline TTS engine)

gTTS (Google Text-to-Speech, optional)

Playsound or pyaudio

Usage:

Run tts_system.py.

Enter the text to convert to speech.

The system reads the text aloud.

3. Building a Speech-to-Text Transcription System with Noise Robustness (RECORDED AUDIO TO TEXT & REAL-TIME SPEECH TO TEXT)
Project Description:
This project features two components: transcribing pre-recorded audio and transcribing real-time speech input. The focus is on building a noise-robust transcription system using filtering and feature extraction techniques.

Key Components:

Denoising audio input using spectral subtraction or bandpass filtering

Transcription of:

Recorded audio files

Live microphone input

Display of transcribed text

Technologies Used:

Python

SpeechRecognition

PyAudio

SciPy (for denoising)

Wave

Usage:

Run recorded_audio_to_text.py for audio file transcription.

Run real_time_transcription.py for live input.

Ensure your mic is configured correctly and ambient noise is minimal.