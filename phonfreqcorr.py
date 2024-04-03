import sys
import itertools

import numpy as np
import pandas as pd

SUBTLEX_FILENAME = "/Users/canjo/data/subtlex/SUBTLEXusfrequencyabove1.csv"
CMU_FILENAME = "/Users/canjo/data/cmudict.0.7a.txt"
CMU_SPLIT = "  "

def read_cmu(filename):
    with open(filename) as infile:
        data = dict(parts for line in infile if len(parts := line.strip().split(CMU_SPLIT)) == 2)
        df = pd.DataFrame(data.items())
    df.columns = ['Word', 'phonemes']
    df['word'] = df['Word'].map(str).map(str.lower)
    return df

def read_freqs(filename):
    df = pd.read_csv(filename)
    df['word'] = df['Word'].map(str).map(str.lower)
    return df

def remove_stress(s):
    return s.strip("0").strip("1").strip("2")

def run(cmu_filename=CMU_FILENAME, freq_filename=SUBTLEX_FILENAME):
    cmu = read_cmu(cmu_filename)
    freq = read_freqs(freq_filename)
    df = pd.merge(cmu[['word', 'phonemes']], freq[['word', 'FREQcount']])
    phonemes = set(itertools.chain.from_iterable(map(str.split, df['phonemes'])))
    phonemes_nostress = set(map(remove_stress, phonemes))
    for phoneme in phonemes_nostress:
        df[phoneme] = df['phonemes'].map(lambda form: phoneme in list(map(remove_stress, form.split())))
    #dfm = pd.melt(df, id_vars=['word', 'phonemes', 'FREQcount'])
    #dfm['logfreq'] = np.log(dfm['FREQcount'])
    #ggplot(dfm[dfm['value']], aes(x='logfreq')) + geom_freqpoly(bins=50) + facet_wrap('~variable') + theme_classic()
    corrs = pd.DataFrame([
        (phoneme, np.corrcoef(np.log(df['FREQcount']), df[phoneme])[0,1])
        for phoneme in phonemes_nostress
    ])
    corrs.columns = ['phoneme', 'corr']
    return df, corrs

if __name__ == '__main__':
    run(*sys.argv[1:]).to_csv(sys.stdout)
    
    
    
    
