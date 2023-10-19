# Simple HiFiGAN

Please cite [the HiFiGAN paper](https://arxiv.org/abs/2010.05646) when using this.

A simple wrapper for the this [HiFiGAN Implementation](
    https://github.com/ming024/FastSpeech2/blob/master/hifigan
). This wrapper just adds an interface to the model.

## Installation

```bash
pip install git+https://github.com/MiniXC/simple_hifigan.git
```

## Usage

```python
from simple_hifigan import Synthesiser

synthesiser = Synthesiser()

# create a mel spectrogram
mel_spectrogram = synthesiser.wav_to_mel(
    some_audio, some_sample_rate
)

# multiple mel spectrograms
mel_spectrograms = synthesiser.wavs_to_mels(
    some_audios, some_sample_rate
)

# mel spectrogram to audio
audio = synthesiser(mel_spectrogram)

