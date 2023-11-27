import torch
import numpy as np

sharp_cats = 'sharp spiky angular jagged hard edgy pointed prickly rugged uneven'.split()
round_cats = 'round circular soft fat chubby curved smooth plush plump rotund'.split()

C = 'bdgktpslhmnx'
V = 'aeiou'
HARD_SOUNDS = set('ptkshixe')
SOFT_SOUNDS = set('bdglumno')
words = [f'{c1}{v1}{c2}{v2}{c1}{v1}' for c1 in C for c2 in C for v1 in V for v2 in V]
words = [w for w in words if (len(set(w) & HARD_SOUNDS) == 0) or (len(set(w) & SOFT_SOUNDS) == 0)]

def is_word_sharp(word):
    return len(set(word) & SOFT_SOUNDS) == 0

def is_word_round(word):
    return len(set(word) & HARD_SOUNDS) == 0

sharp_words = [w for w in words if is_word_sharp(w)]
round_words = [w for w in words if is_word_round(w)]

@torch.no_grad()
def probe(word, model, tokenizer,
        template='a 3D rendering of a {} object',
        cats=['sharp', 'round'],
        add_shaped_word=True,
        add_shaped_cats=False,
        embedding=None,
        cat_embs=None):
    
    assert embedding is None or cat_embs is None

    first = [template.format(f'{word} shaped' if add_shaped_word else word)] if embedding is None else []
    last = [
        template.format(f'{c} shaped' if add_shaped_cats else c)
        for c in cats
    ] if cat_embs is None else []
    prompts = first + last

    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    embs = outputs.text_embeds
    embs /= embs.norm(dim=-1)[:, None]

    if embedding is not None:
        v_mask = embedding
        v_prompts = embs
    elif cat_embs is not None:
        v_mask = embs[0]
        v_prompts = cat_embs
    else:
        v_mask = embs[0] # (512,)
        v_prompts = embs[1:] # (k, 512)
    scores = v_prompts @ v_mask # (k,)
    scores = scores.cpu()
    d = {
        c: s.item()
        for c, s in zip(cats, scores)
    }
    return np.mean(list(d.values()))