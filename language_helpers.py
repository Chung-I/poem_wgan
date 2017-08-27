from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import object
from past.utils import old_div
import collections
import numpy as np
import re

def tokenize_string(sample):
    return tuple(sample.lower().split(' '))

class NgramLanguageModel(object):
    def __init__(self, n, samples, tokenize=False):
        if tokenize:
            tokenized_samples = []
            for sample in samples:
                tokenized_samples.append(tokenize_string(sample))
            samples = tokenized_samples

        self._n = n
        self._samples = samples
        self._ngram_counts = collections.defaultdict(int)
        self._total_ngrams = 0
        for ngram in self.ngrams():
            self._ngram_counts[ngram] += 1
            self._total_ngrams += 1

    def ngrams(self):
        n = self._n
        for sample in self._samples:
            for i in range(len(sample)-n+1):
                yield sample[i:i+n]

    def unique_ngrams(self):
        return set(self._ngram_counts.keys())

    def log_likelihood(self, ngram):
        if ngram not in self._ngram_counts:
            return -np.inf
        else:
            return np.log(self._ngram_counts[ngram]) - np.log(self._total_ngrams)

    def kl_to(self, p):
        # p is another NgramLanguageModel
        log_likelihood_ratios = []
        for ngram in p.ngrams():
            log_likelihood_ratios.append(p.log_likelihood(ngram) - self.log_likelihood(ngram))
        return np.mean(log_likelihood_ratios)

    def cosine_sim_with(self, p):
        # p is another NgramLanguageModel
        p_dot_q = 0.
        p_norm = 0.
        q_norm = 0.
        for ngram in p.unique_ngrams():
            p_i = np.exp(p.log_likelihood(ngram))
            q_i = np.exp(self.log_likelihood(ngram))
            p_dot_q += p_i * q_i
            p_norm += p_i**2
        for ngram in self.unique_ngrams():
            q_i = np.exp(self.log_likelihood(ngram))
            q_norm += q_i**2
        return old_div(p_dot_q, (np.sqrt(p_norm) * np.sqrt(q_norm)))

    def precision_wrt(self, p):
        # p is another NgramLanguageModel
        num = 0.
        denom = 0
        p_ngrams = p.unique_ngrams()
        for ngram in self.unique_ngrams():
            if ngram in p_ngrams:
                num += self._ngram_counts[ngram]
            denom += self._ngram_counts[ngram]
        return old_div(float(num), denom)

    def recall_wrt(self, p):
        return p.precision_wrt(self)

    def js_with(self, p):
        log_p = np.array([p.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_p_m = np.sum(np.exp(log_p) * (log_p - log_m))

        log_p = np.array([p.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_q_m = np.sum(np.exp(log_q) * (log_q - log_m))

        return 0.5*(kl_p_m + kl_q_m) / np.log(2)

def load_dataset(max_length, max_n_examples, tokenize=False, max_vocab_size=2048, data_dir=None, save_data_dir=None):
    print("loading dataset...")

    lines = []

    finished = False

    path = data_dir
    with open(path, 'r') as f:
        for line in f:
            line = line[:-1]
            if tokenize:
                line = tokenize_string(line)
            else:
                line = tuple(line)

            if len(line) > max_length:
                line = line[:max_length]

            lines.append(line + ( ("`",)*(max_length-len(line)) ) )

            if len(lines) == max_n_examples:
                finished = True
                break

    np.random.shuffle(lines)

    import collections
    counts = collections.Counter(char for line in lines for char in line)

    charmap = {'unk':0}
    inv_charmap = ['unk']

    for char,count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))

    if save_data_dir:
        with open(save_data_dir, "w") as fo:
            fo.write("\n".join(filtered_lines))
    for i in range(100):
        print(filtered_lines[i])

    print("loaded {} lines in dataset".format(len(lines)))
    return filtered_lines, charmap, inv_charmap  
# filtered_lines: list of list of chars, with low frequency words replace by 'unk'
# all sequences are either truncated or padded to 32 chars
# charmap: {'unk':0, 'c':1, ... }
# inv_charmap: ['unk', 'c', ...]

def load_tones(max_length, max_n_examples, tokenize=False, max_vocab_size=106, data_dir=None, save_data_dir=None):
    print("loading dataset...")

    lines = []

    finished = False

    path = data_dir
    with open(path, 'r') as f:
        for line in f:
            line = [w for w in line[:-1].split(",")]
            line = tuple(line)
            if len(line) > max_length:
                line = line[:max_length]

            lines.append(line + ( ("`",)*(max_length-len(line)) ) )

            if len(lines) == max_n_examples:
                finished = True
                break

    np.random.shuffle(lines)

    import collections
    counts = collections.Counter(char for line in lines for char in line)

    charmap = {'unk':0}
    inv_charmap = ['unk']

    for char,count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)
    print(charmap)
    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                print("tone {0} not found".format(char))
        filtered_lines.append(tuple(filtered_line))

    if save_data_dir:
        with open(save_data_dir, "w") as fo:
            fo.write("\n".join(filtered_lines))
    for i in range(100):
        print(filtered_lines[i])

    print("loaded {} lines in dataset".format(len(lines)))
    return filtered_lines, charmap, inv_charmap  


def get_mask(charmap, tonemap, char2tone):

    tone_ids = []
    char_tone_map = np.zeros((len(tonemap), len(charmap)), dtype=np.bool)
    #unk_idx = []

    for char, idx in charmap.items():
        try:
            char_all_tones = char2tone[char]
        except:
            #unk_idx.append(idx)
            tone_ids.append(tonemap['unk'])
            continue
        char_tones = next(iter(char_all_tones.values()))
        char_tone = char_tones[0]["聲調"] + char_tones[0]["廣韻同用例"]
        tone_id = tonemap[char_tone]
        tone_ids.append(tone_id)

    indices = np.arange(len(charmap))
    #indices = np.delete(indices, unk_idx)
    char_tone_map[tone_ids, indices] = True

    return char_tone_map

