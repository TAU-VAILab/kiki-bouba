from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers import logging as transformers_logging
from utils import sharp_cats, round_cats, words, probe, is_word_round, is_word_sharp, sharp_words, round_words
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import kendalltau
import torch

def main():
    transformers_logging.set_verbosity_error()

    MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    print(f"Loading model ({MODEL_ID})...")
    model = CLIPTextModelWithProjection.from_pretrained(MODEL_ID).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print("Model loaded")

    ### Geometric Scoring ###

    def probe_s(word):
        return probe(word, model, tokenizer, cats=sharp_cats)
    def probe_r(word):
        return probe(word, model, tokenizer, cats=round_cats)

    scores = {}
    for word in tqdm(words, desc="Calculating geometric scores"):
        scores[word] = (probe_s(word), probe_r(word))

    pdf = pd.DataFrame({'word': scores.keys()}) # pseudoword df
    pdf['is_sharp'] = pdf.word.apply(is_word_sharp)
    pdf['is_round'] = pdf.word.apply(is_word_round)
    assert (pdf.is_sharp ^ pdf.is_round).all(), "Some word is neither sharp nor round."

    pairs = pdf.word.map(scores)
    pdf['s'] = pairs.apply(lambda x: x[0])
    pdf['r'] = pairs.apply(lambda x: x[1])
    pdf['delta'] = pdf.r - pdf.s

    kd = probe_r('kiki') - probe_s('kiki')
    bd = probe_r('bouba') - probe_s('bouba')
    dPkb = (pdf.delta < bd).mean() - (pdf.delta < kd).mean()

    auc = roc_auc_score(pdf.is_round, pdf.delta)
    tau = kendalltau(pdf.is_round, pdf.delta).statistic

    print('Geometric scoring metrics:')
    print(f'\tAUC:\t{auc:.2f}')
    print(f'\tTau:\t{tau:.2f}')
    print(f'\tdPkb:\t{dPkb:.2f}')

    ### Phonetic Scoring ###

    def score_adj(adj):
        return (
            probe(adj, model, tokenizer, cats=round_words, add_shaped_word=True, add_shaped_cats=True)
            - probe(adj, model, tokenizer, cats=sharp_words, add_shaped_word=True, add_shaped_cats=True)
        )
        # Note: add_shaped_word should strictly be false above, but set to true for consistent results
    adf = pd.DataFrame({ # adjective df
        'adj': sharp_cats + round_cats,
        'c': [0] * len(sharp_cats) + [1] * len(round_cats)
    })
    tqdm.pandas(desc="Calculating phonetic scores")
    adf['score'] = adf.adj.progress_apply(score_adj)

    auc_phon = roc_auc_score(adf.c, adf.score)
    tau_phon = kendalltau(adf.c, adf.score).statistic

    print('Phonetic scoring metrics:')
    print(f'\tAUC:\t{auc_phon:.2f}')
    print(f'\tTau:\t{tau_phon:.2f}')


    ### Character-Level Results ###

    def get_charwise(pdf_):
    
        pdf = pdf_.copy()
        
        def gen():
        
            pdf['char'] = pdf.word.apply(lambda x: x[0])
            cdf = pdf.groupby('char').mean(numeric_only=True)
            cdf['delta'] = cdf.r - cdf.s
            yield cdf.sort_values(by='delta')

            pdf['char'] = pdf.word.apply(lambda x: x[1])
            cdf = pdf.groupby('char').mean(numeric_only=True)
            cdf['delta'] = cdf.r - cdf.s
            yield cdf.sort_values(by='delta')
        
        return tuple(gen())

    cons, vows = list(get_charwise(pdf))

    print("Letters sorted by average geometric score:")
    print("\tConsonants:", *list(cons.index))
    print("\tVowels:", *list(vows.index))


    ### Real Word Tests ###

    df = pd.read_csv('data/real_words.csv')
    df['template'] = df.pos.map({
        'adj': 'a 3D rendering of a {} object',
        'noun': 'a 3D rendering of a {} shaped object'
    })
    df['prompt'] = df.apply(lambda row: row.template.format(row.word), axis=1)

    @torch.no_grad()
    def encode(prompts):
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        outputs = model(**inputs)
        E = outputs.text_embeds
        E /= E.norm(dim=-1)[:, None]
        return E
    
    print("Encoding real words...")
    E = encode(df.prompt.tolist())
    SWE = encode([ # sharp pseudowords
        'a 3D rendering of a {} shaped object'.format(word)
        for word in sharp_words
    ])
    RWE = encode([ # round pseudowords
        'a 3D rendering of a {} shaped object'.format(word)
        for word in round_words
    ])
    df['score'] = ((E @ RWE.T) - (E @ SWE.T)).cpu().numpy().mean(axis=-1)

    for x, y in [("Nouns", "noun"), ("Adjectives", "adj")]:
        print(f"{x} sorted by phonetic score:")
        print("\t", *df[df.pos == y].sort_values(by='score', ascending=False).word.head(20).tolist())
        print("\t...")
        print("\t", *df[df.pos == y].sort_values(by='score').word.head(20).tolist())


if __name__ == "__main__":
    main()