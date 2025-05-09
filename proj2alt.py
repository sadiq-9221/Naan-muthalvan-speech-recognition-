from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
import os

# Ensure correct backend for torchaudio to handle flac files
torchaudio.set_audio_backend("sox_io")

# Load the pre-trained model and processor (tokenizer)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Check if a GPU is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load your audio file
audio_path = "C:/Users/user/Downloads/dev-clean (1)/LibriSpeech/dev-clean/2902/9008/2902-9008-0000.flac"

# Check if the file exists
if not os.path.exists(audio_path):
    raise FileNotFoundError(f"The audio file at {audio_path} was not found.")

waveform, sample_rate = torchaudio.load(audio_path)

# Ensure the audio is resampled to 16kHz if it's not already
if sample_rate != 16000:
    waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    sample_rate = 16000  # Update sample rate after resampling

# Ensure correct shape: [batch_size, sequence_length]
waveform = waveform.squeeze()  # Remove unnecessary dimensions if needed

# Process the audio (normalize, extract features) and pass the sampling_rate explicitly
input_values = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values

# Ensure correct shape for the model: [batch_size, sequence_length]
input_values = input_values.squeeze(0)  # Remove the batch dimension (shape should be [seq_length])

# Move input tensor to the same device as the model
input_values = input_values.to(device)

# Perform inference to get logits
with torch.no_grad():
    logits = model(input_values.unsqueeze(0)).logits  # Unsqueeze to add batch dimension back

# Decode the logits to get the transcriptions
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

print("Transcription: ", transcription)
