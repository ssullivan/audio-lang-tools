"""RMVPE: Robust Model for Vocal Pitch Estimation (Wei et al., 2023), a
U-Net and bidirectional GRU on a log-mel spectrogram, giving a pitch
salience over 360 bins (20 cents each) every 10 ms.

Vendored and trimmed to CPU inference from the RVC project's
infer/rmvpe.py (https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI),
MIT License, Copyright (c) 2023 liujing04, 源文雨. The mel filters come
from torchaudio instead of librosa (same HTK mel scale, Slaney norm). The
weights (rmvpe.pt, ~180 MB, MIT) are downloaded to the cache on first use.

    RMVPE().track(x) → (f0 Hz, salience 0–1) per 10 ms frame of 16 kHz audio
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from . import config

WEIGHTS = 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt'


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x):
        return self.conv(x) + (self.shortcut(x) if hasattr(self, 'shortcut') else x)


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super().__init__()
        self.conv = nn.ModuleList([ConvBlockRes(in_channels, out_channels, momentum)]
                                  + [ConvBlockRes(out_channels, out_channels, momentum) for _ in range(n_blocks - 1)])
        self.kernel_size = kernel_size
        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return (x, self.pool(x)) if self.kernel_size is not None else x


class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum))
            in_channels, out_channels = out_channels, out_channels * 2
        self.out_channel = out_channels

    def forward(self, x):
        concat = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            concat.append(t)
        return x, concat


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList([ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)]
                                    + [ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
                                       for _ in range(n_inters - 1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, (3, 3), stride, (1, 1), output_padding=out_padding, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU(),
        )
        self.conv2 = nn.ModuleList([ConvBlockRes(out_channels * 2, out_channels, momentum)]
                                   + [ConvBlockRes(out_channels, out_channels, momentum) for _ in range(n_blocks - 1)])

    def forward(self, x, concat_tensor):
        x = torch.cat((self.conv1(x), concat_tensor), dim=1)
        for conv in self.conv2:
            x = conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            self.layers.append(ResDecoderBlock(in_channels, in_channels // 2, stride, n_blocks, momentum))
            in_channels //= 2

    def forward(self, x, concat):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.encoder = Encoder(in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat = self.encoder(x)
        return self.decoder(self.intermediate(x), concat)


class E2E(nn.Module):
    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        self.fc = nn.Sequential(BiGRU(3 * 128, 256, n_gru), nn.Linear(512, 360), nn.Dropout(0.25), nn.Sigmoid())

    def forward(self, mel):
        x = self.cnn(self.unet(mel.transpose(-1, -2).unsqueeze(1))).transpose(1, 2).flatten(-2)
        return self.fc(x)


class RMVPE:
    RATE, HOP, WIN, N_MELS = 16000, 160, 1024, 128

    def __init__(self):
        path = config.cache('models') / 'rmvpe.pt'
        if not path.exists():
            torch.hub.download_url_to_file(WEIGHTS, str(path))
        self.model = E2E(4, 1, (2, 2))
        self.model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        self.model.eval()
        fb = torchaudio.functional.melscale_fbanks(self.WIN // 2 + 1, 30.0, 8000.0, self.N_MELS, self.RATE,
                                                   norm='slaney', mel_scale='htk')
        self.mel_basis = fb.T  # [mels, freqs]
        self.window = torch.hann_window(self.WIN)
        cents = 20 * np.arange(360) + 1997.3794084376191
        self.cents = np.pad(cents, (4, 4))

    def mel(self, audio: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(audio, self.WIN, self.HOP, self.WIN, window=self.window, center=True, return_complex=True).abs()
        return torch.log(torch.clamp(self.mel_basis @ spec, min=1e-5))

    def track(self, x: np.ndarray):
        with torch.inference_mode():
            mel = self.mel(torch.from_numpy(x).float().unsqueeze(0))
            n = mel.shape[-1]
            mel = F.pad(mel, (0, 32 * ((n - 1) // 32 + 1) - n))
            sal = self.model(mel)[0, :n].numpy()
        # Local weighted average of cents around the peak bin.
        center = np.argmax(sal, axis=1) + 4
        padded = np.pad(sal, ((0, 0), (4, 4)))
        idx = center[:, None] + np.arange(-4, 5)[None, :]
        w = np.take_along_axis(padded, idx, axis=1)
        cents = (w * self.cents[idx]).sum(1) / np.maximum(w.sum(1), 1e-9)
        return 10 * 2 ** (cents / 1200), sal.max(axis=1)
