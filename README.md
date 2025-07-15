<p align="center" width="100%">
<a target="_blank"><img src="img/llama_logo.png" alt="IVCR" style="width: 40%; min-width: 150px; display: block; margin: auto;"></a>
</p>
<h2 align="center"> IVCR-200K: A Large-Scale Multi-turn Dialogue
Benchmark for Interactive Video Corpus Retrieval</h2>

## Model Architecture
<p align="center" width="100%">
<a target="_blank"><img src="img/tone.png" alt="Video-LLaMA" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

## Introduction
**IVCR** enables multi-turn, conversational, realistic interactions
between the user and the retrieval system. The main contributions are summarized as follows: i)-To the best of our knowledge, this is the first
70 work to introduce the “interactive” video corpus retrieval task (IVCR) , which effectively aligns
71 users’ multi-turn behavior in real-world scenarios and significantly enhances user experience. 

ii)-We
72 introduce a dataset and accompanying framework. Notably, the IVCR-200K dataset is a high73
quality, bilingual, multi-turn, conversational, and abstract semantic dataset designed to support video
74 and moment retrieval. 

## Example Outputs
<p float="left">
    <img src="img/a.png" style="width: 100%; margin: auto;">
</p>

## Dataset
We introduce <a href="https://drive.google.com/drive/folders/1ScmHPvpZP_BiiibFgze1WWW3T9I2K15m?usp=sharing">IVCR-200K</a>, a bilingual, multi-turn, conversational, abstract semantic high-quality dataset that supports video retrieval and even moment retrieval.

## Usage
#### Enviroment Preparation 
create a conda environment
```
conda create -n envname pyhton=3.11
conda activate envname
pip install -r requirements.txt
```
### checkpoints
make sure you have obtained the following checkpoints
#### Pre-trained Image Encoder (EVA ViT-g)
```bash
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
```
#### Pre-trained Image Q-Former (InstructBLIP Q-Former)
```bash
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth
```
#### Pre-trained Language Decoder (LLaMA-2-7B) and Video Encoder (Video Q-Former of Video-LLaMA)

Use `git-lfs` to download weights of [Video-LLaMA (7B)](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/tree/main):
```bash
git lfs install
git clone https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned
```

#### Instruct-tuned [TimeChat-7B](https://huggingface.co/ShuhuaiRen/TimeChat-7b)
```bash
git lfs install
git clone https://huggingface.co/ShuhuaiRen/TimeChat-7b
```

## How to Run
### Data Preprocessing
```
download IVCR_200K
run ./convert_dataset.ipynb 
```
### Tuning
```
python train.py --cfg-path ./train_configs/stage2_finetune_IVCR.yaml
```

### Evaluating
```
python evaluate.py 
```
```
python ./metrics/eval.py
```