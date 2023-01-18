# TaskRes
The official implementation of [*Task Residual for Tuning Vision-Language Models*](https://arxiv.org/abs/2211.10277).

The proposed Task Residual Tuning (TaskRes) is a new paradigm for tuning vision-language models (VLMs), which directly tunes the text-based classifier weights, without the need of heavy text encoders for prompt updates or carefully designed adapters.

![image](./images/taskres.png)

## Installation
This repository requires install the environment and datasets:
- follow [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) and PyTorch.
- run `pip install -r requirements.txt` under `TaskRes/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated).
- follow [DATASETS.md](DATASETS.md) to install the datasets.

*PS: You can also follow [CoOp](https://github.com/KaiyangZhou/CoOp) to complete the installation.*

## Usage
We present basic usage here.

(a) Train regular TaskRes:
- see [train_regular.sh](train_regular.sh) to run regular TaskRes (i.e., using regular base).

(b) Train enhanced TaskRes:
- Download [enhanced bases](https://drive.google.com/drive/folders/1_ehtvBRWbbcYZRTAcvtCyUTD_tL4GUiV?usp=share_link) to `TaskRes/`.
- See [train_enhance.sh](train_enhance.sh) to run enhanced TaskRes (i.e., using enhanced base).

(c) Test domain generalization:
- See [test_dg.sh](test_dg.sh) to run enhanced TaskRes (i.e., using enhanced base).

*PS: Refer to [CoOp](https://github.com/KaiyangZhou/CoOp) for more usage.*

## Acknowledgment
This repository is mainly based on Kaiyang Zhou's repository [CoOp](https://github.com/KaiyangZhou/CoOp) code base. We sincerely thank Kaiyang for his awesome code base.
