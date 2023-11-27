from utils import words
import os
from tqdm.auto import tqdm
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers import logging as transformers_logging
import pandas as pd
import torch
from glob import glob
from PIL import Image


def main():

    IMAGE_DIR = 'generated_images'
    OUTPUT_DIR = 'embeddings'
    BSZ = 2 ** 9

    if not os.path.exists(OUTPUT_DIR):
        print("Making output embedding directory:", OUTPUT_DIR)
        os.mkdir(OUTPUT_DIR)

    transformers_logging.set_verbosity_error()

    MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    print(f"Loading model ({MODEL_ID})...")
    model = CLIPVisionModelWithProjection.from_pretrained(MODEL_ID).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Model loaded")

    words_kb = words + ['kiki', 'bouba']
    words_kb_set = set(words_kb)
    df = pd.DataFrame({'fn': glob(f'{IMAGE_DIR}/*/*.jpg')})
    df['label'] = df.fn.str.extract(f'([^/]*)/[^/]*$')[0]
    df = df[df.label.isin(words_kb_set)].copy()
    seen_words = set(df.label.unique())
    assert len(words_kb_set - seen_words) == 0, "Missing images for some pseudowords"

    for label, subdf in tqdm(df.groupby('label'), desc=f"Embedding pseudowords (bsz={BSZ})"):
        fn = f'{OUTPUT_DIR}/{label}.pt'
        if not os.path.exists(fn):
            subdf = subdf.reset_index(drop=True)
            subdf['batch'] = subdf.index // BSZ
            out = []
            for _, ssdf in subdf.groupby('batch'):
                imgs = [Image.open(fn) for fn in ssdf.fn]
                with torch.no_grad():
                    inputs = processor(images=imgs, return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    embs = outputs.image_embeds.cpu()
                    out.append(embs)
            E = torch.vstack(out)
            torch.save(E, fn)


if __name__ == "__main__":
    main()