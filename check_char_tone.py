import argparse
import json
import os
import re

parser = argparse.ArgumentParser(description="gan_language.py")
parser.add_argument('-char_info', required=True,
                    help='character info path.')
parser.add_argument('-model', required=True,
                    help='model path.')
parser.add_argument('-epoch', required=True,
                    help='epoch of the output')

opt = parser.parse_args()

def get_match_rate(chars, tones, char2tone):

    tone_ids = []
    unk_idx = []
    num_chars = 0
    correct = 0
    for cl, tl in zip(chars, tones):
        cl = re.findall("(unk|[^A-Za-z0-9])", cl)
        for c, t in zip(cl, tl):
            num_chars += 1
            try:
                char_all_tones = char2tone[c]
            except:
                continue
            char_all_tones.values()
            char_tones = []
            pairs = iter(char_all_tones.values())
            while True:
                try:
                    c_ts = next(pairs)
                    char_tone = c_ts[0]["聲調"] + c_ts[0]["廣韻同用例"]
                    char_tones.append(char_tone)
                except:
                    break
            if t in char_tones:
                correct += 1
    return correct / num_chars

with open(opt.char_info) as f:
    char2tone = json.load(f)
with open(os.path.join(opt.model, "samples_{}.txt".format(opt.epoch))) as f:
    chars = f.read().splitlines()
with open(os.path.join(opt.model, "tones_{}.txt".format(opt.epoch))) as f:
    tones = [line.split(",") for line in f.read().splitlines()]


rate = get_match_rate(chars, tones, char2tone)
print(rate)
