
<div align="center">

# *SFT Memorizes, RL Generalizes*: <br>A Comparative Study of Foundation Model Post-training

<p>
    <img src="assets/teaser.png" alt="Cambrian" width="500" height="auto">
</p>



<a href="https://arxiv.org/abs/2501.17161v1" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-SFT vs RL-red?logo=arxiv" height="25" />
</a>
<a href="https://tianzhechu.com/SFTvsRL/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-tianzhechu.com/SFTvsRL-blue.svg" height="25" />
</a>
<a href="https://huggingface.co/collections/tianzhechu/sftvsrl-models-and-data-6797ba6de522c7de7fcb80bah" target="_blank">
    <img alt="HF Model: Cambrian-1" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-Checkpoints&Data-ffc107?color=ffc107&logoColor=white" height="25" />
</a>


<div style="font-family: charter; text-align: center; margin: 0 auto;">
                    <a href="https://tianzhechu.com/" class="author-link" target="_blank">Tianzhe Chu*</a> &emsp;
                    <a href="https://yx-s-z.github.io/" class="author-link" target="_blank">Yuexiang Zhai*</a> &emsp;
                    <a href="https://jihanyang.github.io/" class="author-link" target="_blank">Jihan Yang</a> &emsp;
                    <a href="https://tsb0601.github.io/petertongsb/" class="author-link" target="_blank">Shengbang Tong</a> &emsp;
                    <br>
                    <a href="https://www.sainingxie.com/" class="author-link" target="_blank">Saining Xie</a> &emsp;
                    <a href="https://webdocs.cs.ualberta.ca/~dale/" class="author-link" target="_blank">Dale Schuurmans</a> &emsp;
                    <a href="https://cs.stanford.edu/~quocle/" class="author-link" target="_blank">Quoc V. Le</a> &emsp;
                    <a href="https://people.eecs.berkeley.edu/~svlevine/" class="author-link" target="_blank">Sergey Levine</a> &emsp;
                    <a href="https://people.eecs.berkeley.edu/~yima/" class="author-link" target="_blank">Yi Ma</a> &emsp;
</div>
<br>
</div>


> *Misc: We prompt DALL-E 3 via "Conceptual figure of 'SFT Memorizes, RL Generalizes', with trendlines and style of Hong Kong" but somehow skycrapters dominate the picture...*

## Release
- [02/8/25] We add [SFT scripts](sft/) and text-only [SFT data](https://huggingface.co/datasets/tianzhechu/SFTvsRL_Data). Keep updating in the future.
- [01/28/25] Excited to shout out our paper *SFT Memorizes, RL Generalizes*! We release the environments, training scripts, evaluation scripts, SFT data, and initial checkpoints.

## Installation

### Prepare
Our codebase is tested on H800 servers with <code>python 3.13.0</code> <code>torch 2.5.1+cu124</code>.

1. Clone this repository and navigate to into the codebase
```Shell
git clone https://github.com/LeslieTrue/SFTvsRL.git
cd SFTvsRL
```

2. Install Packages
```Shell
conda create -n SFTvsRL python==3.13 -y
conda activate SFTvsRL
pip install -r requirements.txt
cd gym
pip install -e . # install gym environment
cd ..
```

### Download Initial Checkpoints (Optional)
We instantiate RL experiments on top of SFT initialized checkpoints to guarantee model's basic instruction following capabilities. We provide all 4 initial checkpoints for \{GeneralPoints, V-IRL\}X\{Language (-L), Vision-Language (-VL)\}. 
```Shell
huggingface-cli download tianzhechu/GP-L-Init --local-dir YOUR_LOCAL_DIR
huggingface-cli download tianzhechu/GP-VL-Init --local-dir YOUR_LOCAL_DIR
huggingface-cli download tianzhechu/VIRL-L-Init --local-dir YOUR_LOCAL_DIR
huggingface-cli download tianzhechu/VIRL-VL-Init --local-dir YOUR_LOCAL_DIR
```
It's optional to download these checkpoints via huggingface CLI. You may directly specify <code>repo_name</code> as <code>CKPT_NAME</code> in shell scripts.

## Getting Started

1. Install packages and prepare the initial checkpoints (optional).
   - Check [here](https://huggingface.co/collections/tianzhechu/sftvsrl-models-and-data-6797ba6de522c7de7fcb80ba) to download initial checkpoints for all 4 training experiments.
   - You may train your own initial checkpoints following [instructions here](sft/README.md).
   - We use [Llama-3.2-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) for all our experiments. Other models might not need SFT initialization and welcome to explore~
2. Launch RL experiments (PPO).
   - For GeneralPoints, please use execute the following scripts:
     - Language only: <code>bash scripts/gp_training/language_train.sh</code>
     - With vision: <code>bash scripts/gp_training/vl_train.sh</code>
     - Edit training configs either in shell scripts or <code>rl/configs/llama_gp_*.yaml</code>
   - For V-IRL, please do the following steps:
     - First, download data from [here](https://huggingface.co/datasets/tianzhechu/SFTvsRL_Data).
     - Then, specify paths in training shell scripts
       - <code>STREETVIEWS=YOUR_PATH/nyc_1k_routes/street_views/</code>
       - <code>GPS_TO_PANO=YOUR_PATH/nyc_1k_routes/gps_pano_mapping.pkl</code>
       - <code>ROUTE_INFO=YOUR_PATH/nyc_1k_routes/route_infos.json</code>
     - Finally, start training
       - Language only: <code>bash scripts/virl_training/language_train.sh</code>
       - With vision: <code>bash scripts/virl_training/vl_train.sh</code>
       - Edit training configs either in shell scripts or <code>rl/configs/llama_virl_*.yaml</code>
3. Evaluate RL checkpoints after training.
    - We have a series of evaluation scripts:
      - <code>scripts/gp_evaluation/*.sh</code>: evaluate GeneralPoints
      - <code>scripts/virl_evaluation/*.sh</code>: evaluate V-IRL
      - <code>scripts/recog_evaluation/*.sh</code>: evaluate GeneralPoints recognition
    - Please modify <code>CKPT_NAME</code> in these shell scripts.

\*\* Note that our shell scripts support slurm clusters if launched via <code>sbatch scripts/\*/\*.sh</code>. Reproducing our training experiments require a node of 8 gpus with memory of 80GB each. 
## Citation

If you find this project useful for your research and applications, please cite using this BibTeX:
```bibtex
@misc{chu2025sftmemorizesrlgeneralizes,
      title={SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training}, 
      author={Tianzhe Chu and Yuexiang Zhai and Jihan Yang and Shengbang Tong and Saining Xie and Dale Schuurmans and Quoc V. Le and Sergey Levine and Yi Ma},
      year={2025},
      eprint={2501.17161},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2501.17161}, 
}
```

## Acknowledgement

- [RL4VLM](https://github.com/RL4VLM/RL4VLM): We start our codebase from Simon's amazing project.
- [Llama-3.2-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct): We instantiate our experiments on top of this model.
- [Llama-3.2-Vision-Finetune](https://github.com/2U1/Llama3.2-Vision-Finetune): Our SFT code is modified from early version of this repository.
- [V-IRL: Grounding Virtual Intelligence in Real Life](https://virl-platform.github.io/): We adopt this fantastic environment.

