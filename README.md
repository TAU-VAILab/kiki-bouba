# Kiki or Bouba? Sound Symbolism in Vision-and-Language Models (NeurIPS 2023)

[Project Page](https://kiki-bouba.github.io/) | [Paper](https://arxiv.org/abs/2310.16781)

This is the official repository for the paper: *Morris Alper and Hadar Averbuch-Elor (2023). Kiki or Bouba? Sound Symbolism in Vision-and-Language Models. NeurIPS 2023*

## Code

All code has been tested with Python 3.9 on a single NVIDIA A5000 GPU. After installing the required libraries (`pip install -r requirements.txt`) you may reproduce our results as follows:

* CLIP:
  * `python src/clip_tests.py`
* Stable Diffusion (SD):
  * `python src/sd_generate_images.py`
  * `python src/sd_embed_images.py`
  * `python src/sd_tests.py`

The SD scripts must be run in order. Note that SD generations use random seeds and thus results may differ slightly from those reported in our paper due to randomness.

## Licence

We release our code under the [MIT license](https://opensource.org/license/mit/).

## Citation

If you find this code or our data helpful in your research or work, please cite the following paper.
```
@InProceedings{alper2023kiki-or-bouba,
    author    = {Morris Alper and Hadar Averbuch-Elor},
    title     = {Kiki or Bouba? Sound Symbolism in Vision-and-Language Models},
    booktitle = {Proceedings of Advances in Neural Information Processing Systems (NeurIPS)},
    year      = {2023}
}
```