import json
from pathlib import Path
import os

import torch
import torchaudio
import torchaudio.transforms as AT
from librosa.filters import mel as librosa_mel
import numpy as np

from simple_hifigan.models import Generator


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Synthesiser:
    def __init__(
        self,
        device="cpu",
        model="universal",
    ):
        self.device = device
        with open(Path(__file__).parent / "data" / "config.json", "r") as f:
            config = json.load(f)
        config = AttrDict(config)
        vocoder = Generator(config)
        ckpt = torch.load(
            Path(__file__).parent / "data" / f"generator_{model}.pth.tar",
            map_location=torch.device(device),
        )
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(self.device)
        self.vocoder = vocoder

        self.mel_fn = AT.Spectrogram(
            n_fft=config.n_fft,
            win_length=config.win_size,
            hop_length=config.hop_size,
            pad=0,
            window_fn=torch.hann_window,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.mel_basis = librosa_mel(
            sr=config.sampling_rate,
            n_fft=config.n_fft,
            n_mels=config.num_mels,
            fmin=config.fmin,
            fmax=config.fmax,
        )
        self.mel_basis = torch.from_numpy(self.mel_basis).float()

    def _drc(self, x, C=1, clip_val=1e-7):
        return torch.log(torch.clamp(x, min=clip_val) * C)

    def wav_to_mel(self, wav, sr=None):
        if isinstance(wav, str):
            wav, sr = torchaudio.load(wav)
        elif isinstance(wav, Path):
            wav, sr = torchaudio.load(str(wav))
        elif isinstance(wav, torch.Tensor):
            wav = wav.unsqueeze(0)
        elif isinstance(wav, list):
            wav = torch.tensor(wav).unsqueeze(0)
        elif isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).unsqueeze(0)
        else:
            raise ValueError("wav must be either a path or a torch.Tensor")

        if sr is None:
            raise ValueError("sample rate must be specified if wav is not a path")

        wav = wav.to(self.device)
        if sr != 22050:
            wav = torchaudio.transforms.Resample(sr, 22050)(wav)

        mel = self.mel_fn(wav)
        mel = torch.sqrt(mel)
        mel = torch.matmul(self.mel_basis, mel)
        mel = self._drc(mel)
        return mel

    def wavs_to_mel(self, wavs, sr=None):
        if isinstance(wavs, str):
            wavs, sr = torchaudio.load(wavs)
        elif isinstance(wavs, Path):
            wavs, sr = torchaudio.load(str(wavs))
        elif isinstance(wavs, torch.Tensor):
            pass
        elif isinstance(wavs, list):
            wavs = torch.tensor(wavs)
        elif isinstance(wavs, np.ndarray):
            wavs = torch.from_numpy(wavs)
        else:
            raise ValueError("wav must be either a path or a torch.Tensor")

        if sr is None:
            raise ValueError("sample rate must be specified if wav is not a path")

        wavs = wavs.to(self.device)
        if sr != 22050:
            wavs = torchaudio.transforms.Resample(sr, 22050)(wavs)

        mels = self.mel_fn(wavs)
        mels_mask = mels.sum(dim=1) != 0
        mels = torch.sqrt(mels)
        mels = torch.matmul(self.mel_basis, mels)
        mels = self._drc(mels)
        return mels, mels_mask

    def __call__(self, mel):
        if isinstance(mel, np.ndarray):
            mel = torch.from_numpy(mel).float()
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        mel = mel.to(self.device)
        result = self.vocoder(mel).squeeze(1).cpu().detach().numpy()
        result = (result * 32768.0).astype("int16")

        return result
