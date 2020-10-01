import os

import requests
import torch
from fairseq.models.wav2vec import Wav2VecModel


class Wav2VecFeat():
    def __init__(self, wav2vec_dir=os.path.join('audio_classifier', 'wav2vec'), 
                 weights_url='https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt',
                 weights_fn='wav2vec_large.pt'):
        self.wav2vec_dir = wav2vec_dir
        self.weights_url = weights_url
        self.weights_fn = weights_fn
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if not self.weights_exist():
            self.download_weights()
        self.load_weights()

    def weights_exist(self):
        return os.path.exists(os.path.join(self.wav2vec_dir, self.weights_fn))

    def download_weights(self):
        print(f'Downloading weights: {self.weights_fn}')
        with requests.get(self.weights_url, stream=True) as r:
            r.raise_for_status()
            with open(os.path.join(self.wav2vec_dir, self.weights_fn), 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    print('.', end='')

    def load_weights(self):
        cp = torch.load(os.path.join(self.wav2vec_dir, self.weights_fn), map_location=self.device)
        self.model = Wav2VecModel.build_model(cp['args'], task=None)
        self.model.load_state_dict(cp['model'])
        self.model.eval()

    def extract_feature(self, wav_input_16khz):
        z = self.model.feature_extractor(wav_input_16khz)
        c = self.model.feature_aggregator(z)
        return c
