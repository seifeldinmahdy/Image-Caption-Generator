# Image Caption Generator

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-EE4C2C?logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.2+-000000?logo=flask&logoColor=white)

An adversarially-trained image captioning model that generates natural-language descriptions of
photographs, trained on the MS COCO 2014 captions dataset and served through a small Flask web app
with a drag-and-drop interface.

A ResNet-101 encoder turns an image into a feature vector, a two-layer LSTM decoder generates the
caption with beam search, and a pair of discriminators (CNN- and RNN-based) push the generator
towards captions that are hard to distinguish from human-written ones.

```
Input image  ──►  ResNet-101 encoder  ──►  LSTM decoder + beam search  ──►  "a man riding a wave on a surfboard"
                                                   ▲
                                        CNN + RNN discriminators
                                        (adversarial training only)
```

## Contents

- [How it works](#how-it-works)
- [Results](#results)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Running the web app](#running-the-web-app)
- [Training your own model](#training-your-own-model)
- [Limitations](#limitations)
- [References](#references)

## How it works

**Encoder** — A ResNet-101 pre-trained on ImageNet, with its classification head removed and its
weights frozen. The 2048-d pooled feature is projected to a 512-d embedding through a linear layer
with batch normalisation and dropout.

**Decoder** — A two-LSTM-cell "top-down" decoder. At each timestep the first cell consumes the
previous word embedding concatenated with the image feature, and the second cell turns that hidden
state into a distribution over the 8,853-token vocabulary. Captions are generated with beam search
(beam size 5, maximum length 50 tokens).

**Adversarial training** — Two discriminators score whether an image–caption pair looks real: a CNN
discriminator over multiple kernel widths, and an RNN discriminator. Their ensemble output is
combined with a CIDEr-D reward and fed back to the generator through a policy-gradient (SCST-style)
loss, alongside the usual cross-entropy objective. Mismatched image–caption pairs are used as
additional negatives.

**Training schedule** — 10 epochs of maximum-likelihood pre-training for the generator, 5 epochs of
discriminator pre-training, then 3 epochs of joint adversarial training.

**Data** — MS COCO 2014 captions. The reference run used a fixed 30% subset (seed 42): 24,834
training images and 12,151 validation images. The vocabulary keeps words appearing at least 5 times,
giving 8,853 tokens including `<pad>`, `<start>`, `<end>` and `<unk>`.

## Results

Validation scores from the reference run (30% COCO subset, batch size 128, Adam at 5e-3, Tesla T4):

| Epoch | BLEU-4 | METEOR | ROUGE-L |
| ----- | ------ | ------ | ------- |
| 1     | 0.2449 | 0.4282 | 0.4652  |
| 2     | 0.2387 | 0.4313 | 0.4639  |
| 3     | 0.2454 | 0.4384 | 0.4712  |

The released weights come from the epoch-1 checkpoint (BLEU-4 0.2449, METEOR 0.4282,
ROUGE-L 0.4652).

CIDEr and SPICE are wired up in the evaluation code but did not produce scores in the reference run —
SPICE needs a working Java runtime plus Stanford CoreNLP, which failed in the Colab environment, and
both metrics were reported as 0.0. Treat them as not measured rather than as genuine zeros.

Sample validation captions:

| Generated caption                                    | A human reference                                              |
| ---------------------------------------------------- | -------------------------------------------------------------- |
| a man swinging a tennis racket on a tennis court .   | a man taking a swing at a tennis ball                          |
| a man riding a wave on a surfboard .                 | a young surfer in a wetsuit surfs a small wave .               |
| a close up of a street sign on a pole .              | traffic and street signs sit on the pole                       |
| a plate of food on a table .                         | a microwave and a cup of coffee on a table .                   |
| a group of people standing on a tennis court .       | a boy on a baseball field hits a baseball                      |

The last row is a representative failure: the model falls back on a frequent COCO scene ("tennis
court") when the sport is ambiguous.

## Project structure

```
.
├── app.py                          Flask app: upload endpoint and caption inference
├── model.py                        ImageEncoder and TopDownAttention used at serving time
├── templates/
│   └── index.html                  Drag-and-drop front end
├── notebooks/
│   ├── train_captioner.ipynb       Full pipeline: data download, training, evaluation, export
│   └── build_vocab.ipynb           Builds vocab.pkl from the COCO caption annotations
├── vocab.pkl                       Serialised word2idx / idx2word mappings (8,853 tokens)
├── requirements.txt                Runtime dependencies for the web app
└── requirements-train.txt          Dependencies for the notebooks
```

Not tracked in this repository: the COCO dataset (`data/`, ~20 GB, downloaded by the training
notebook) and the model weights (`final_model_weights.pth`, see below).

## Getting started

Requires Python 3.10 or newer.

```bash
git clone https://github.com/seifeldinmahdy/Image-Caption-Generator.git
cd Image-Caption-Generator

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Model weights

`app.py` expects a checkpoint named `final_model_weights.pth` in the repository root. It is not
committed here because of its size. Produce it by running `notebooks/train_captioner.ipynb` to the
end — the final cell writes the file and downloads it — then drop it next to `app.py`.

The checkpoint is a dictionary with `encoder_state_dict`, `generator_state_dict`,
`cnn_discriminator_state_dict` and `rnn_discriminator_state_dict`; only the first two are needed for
inference.

## Running the web app

```bash
python app.py
```

Open <http://127.0.0.1:5000> and drop an image onto the upload area. Uploads are capped at 16 MB and
are held in memory only — nothing is written to disk. The caption is generated with beam search and
returned as JSON.

If the weights cannot be loaded the app prints the error and exits instead of starting, so a missing
or mismatched checkpoint fails loudly rather than serving nonsense.

**API** — `POST /upload` with a multipart form field named `file`:

```bash
curl -F "file=@photo.jpg" http://127.0.0.1:5000/upload
# {"caption": "a man riding a wave on a surfboard .", "success": true}
```

The development server is for local use only; put it behind a production WSGI server such as Gunicorn
if you deploy it.

## Training your own model

Training was done on Colab with a T4 GPU, and the notebook is written for that environment.

```bash
pip install -r requirements-train.txt
```

1. Open `notebooks/train_captioner.ipynb`. The first cells download and extract the MS COCO 2014
   images and annotations into `data/coco/` (roughly 20 GB, so expect a long first run).
2. Adjust the subset fraction, batch size, learning rate and epoch counts in the configuration cell
   near the end. The defaults reproduce the run in the results table.
3. Run through training and evaluation, then execute the final cell to export
   `final_model_weights.pth`.
4. If you change the vocabulary threshold, regenerate `vocab.pkl` with
   `notebooks/build_vocab.ipynb` (run it from the `notebooks/` directory; it reads the annotations
   from `../data/` and writes the pickle to the repository root) so that the app and the checkpoint
   agree on vocabulary size.

## Limitations

- Trained on 30% of COCO for 3 adversarial epochs on a single T4, so scores sit below published
  results for this family of architectures.
- The decoder attends to a single pooled image vector rather than spatial or bottom-up region
  features, which limits how well it grounds fine-grained detail.
- Captions reflect COCO's distribution: everyday scenes, objects and sports. Out-of-domain images
  (diagrams, screenshots, artwork) tend to be described with a plausible-sounding but wrong caption.
- CIDEr and SPICE evaluation remains unverified in this environment.

## References

- Anderson et al., [Bottom-Up and Top-Down Attention for Image Captioning and VQA](https://arxiv.org/abs/1707.07998) (CVPR 2018)
- Xu et al., [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) (ICML 2015)
- Rennie et al., [Self-Critical Sequence Training for Image Captioning](https://arxiv.org/abs/1612.00563) (CVPR 2017)
- Dai et al., [Towards Diverse and Natural Image Descriptions via a Conditional GAN](https://arxiv.org/abs/1703.06029) (ICCV 2017)
- Lin et al., [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312) (ECCV 2014)
