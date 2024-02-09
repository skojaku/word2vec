import numpy as np
import pandas as pd


class Word2VecDataset:
    def __init__(self, input_text, window=5):
        self.input_text = input_text
        self.vocab = np.unique(input_text)
        self.window = window
        self.n_samples = 0

    def get_vocab(self):

        return self.vocab

    def __len__(self):

        pass

    def __getitem__(self, _):
        if self._idx == self.n_samples:
            self._generate_samples()
            self._idx = 0

        retval = self.center_words[self._idx], self.context_words[self._idx]
        self._idx += 1
        return retval

    def _generate_samples(self):


        
        return 1


class NegativeSampling:
    def __init__(self, input_text):
        self.vocab, self.freq = np.unique(self.input_text, return_counts=True)
        self.p = self.freq / np.sum(self.freq)

    def sample(self):
        return np.random.choice(self.vocab, p=self.p)
