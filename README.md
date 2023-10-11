# RationaleCL

Open-Source code for EMNLP 2023 paper: *[Rationale-Enhanced Language Models are Better Continual Relation Learners](https://arxiv.org/abs/2310.06547)*

## Environment

- Python: 3.7.11
- Torch: 1.3.11+cu117

```
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

## Dataset

- Create a new folder `datasets`.
- Download the dataset FewRel and TACRED from [Google Drive](https://drive.google.com/drive/folders/1z0aYk2HwfzYan2v5jyPxmie3YuXHsm_J?usp=sharing) and place them in `datasets`.
- We construct $5$  task sequences for each dataset.
  - For FewRel and TACRED, our 5 task sequences are the same as the same as [RP-CRE](https://aclanthology.org/2021.acl-long.20/), [CRL](https://aclanthology.org/2022.findings-acl.268/) and [ACA](https://aclanthology.org/2022.emnlp-main.420/).
  - Please refer to `generate_tasks.py` for more details.
- Run `python generate_tasks.py +task_args=FewRel/TACRED` to construct 5 task sequences for each task used in our paper.

## Run

* make sue you have the following file structure:

```
├── bash
│   ├── FewRel
│   └── TACRED
├── configs
│   ├── default.yaml
│   ├── model_args
│   ├── task_args
│   └── training_args
├── data
│   ├── BaseData.py
│   ├── __init__.py
│   └── TACREDFewRel.py
├── datasets
│   ├── FewRel
│   └── TACRED
├── generate_tasks.py
├── main.py
├── model
│   ├── CLT5.py
│   ├── __init__.py
├── README.md
├── requirements.txt
├── sampled_data
│   ├── FewRel
│   └── TACRED
├── train
│   ├── DefaultCollator.py
│   ├── DefaultEvaluate.py
│   ├── DefaultHyperTrain.py
│   ├── MTTrain.py
│   ├── __init__.py
└── utils
    ├── __init__.py
    └── utils.py
```

* Get OpenAI key from OpenAI and fill in your own key in the function `send_request` of `utils/utils.py`.
* The code can be run using the following command:

```
bash bash/[dataset]/mt.sh
    - dataset: the dataset name, e.g.,:
        - FewRel/TACRED
```

For example,

```
bash bash/FewRel/mt.sh
```

 The model we used for our experiments was gpt-3.5-turbo-0301, but this model has now been deprecated. As a result, the generated contrastive rationales may be somewhat different, potentially leading to some variations in the final results.

## Citation

If you find this repo useful, please cite us.

```bibtex
@misc{xiong2023rationaleenhanced,
    title={Rationale-Enhanced Language Models are Better Continual Relation Learners},
    author={Weimin Xiong and Yifan Song and Peiyi Wang and Sujian Li},
    year={2023},
    eprint={2310.06547},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
