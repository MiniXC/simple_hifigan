from simple_hifigan import Synthesiser
import torchaudio
import torch

synthesiser = Synthesiser()

# create a mel spectrogram
mel_spectrogram = synthesiser.wav_to_mel("tests/audio.wav")

audio = synthesiser(mel_spectrogram)

torchaudio.save("tests/audio_generated.wav", torch.tensor(audio), 22050)
