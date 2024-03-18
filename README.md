# Matrix-Transformation Based Low-Rank Adaptation (MTLoRA): A Brain-Inspired Method for Parameter-Efficient Fine-Tuning

Welcome to MTLoRA.

**Matrix-Transformation Based Low-Rank Adaptation (MTLoRA): A Brain-Inspired Method for Parameter-Efficient Fine-Tuning** <br>
*Yao Liang, Yuwei Wang, Yi Zeng* <br>
Paper: https://arxiv.org/abs/2403.07440 <br>

This paper explores the enhancement of fine-tuning techniques in Large Pretrained Language Models (LPLMs), crucial for improving performance across various tasks while efficiently managing computational resources. Our proposed method, MTLoRA, introduces a novel reparameterization approach inspired by the brain's geometric structure, applying a transformation matrix to adjust the task-specific parameter matrix. This innovation aims to mimic the brain's geometric structural influence on functionality, thereby boosting model performance in downstream tasks. Our evaluations demonstrate notable improvements in both Natural Language Understanding and Generation tasks.

## Repository Overview

This repository includes multiple directories:
* The [examples/](examples) This directory contains tasks of two major categories: natural language understanding and natural language generation.
* The [transformation_matrix/](transformation_matrix) This directory contains four types of structures for transformation matrices.

When using a particular structure of MTLoRA, simply replace the ```transformation_matrix\CTCM\xtuning_layers.py``` file from that structure into the following directory:
```console
1. examples\NLU\src\xtuninglib\xtuning_layers.py
2. examples\NLU\src\transformers\models\roberta\xtuninglib\xtuning_layers.py
3. examples\NLG\src\xtuninglib\xtuning_layers.py
```

# Natural Language Understanding Tasks

```console
examples/NLU/
```
## Getting Started

### Set up and initiate a conda environment.

```console
conda env create -f environment.yml
```
### Install the necessary dependencies.

```console
pip install -e .
```
### Begin the experimental procedures.
```console
roberta_base_cola.sh
roberta_base_mnli.sh
roberta_base_mrpc.sh
roberta_base_qnli.sh
roberta_base_qqp.sh
roberta_base_rte.sh
roberta_base_sst2.sh
roberta_base_stsb.sh
```
For MRPC, RTE, and STSB tasks, begin with the MTLoRA-adapted MNLI checkpoint and modify the file path in the shell script as required.


# Natural Language Generation Tasks

```console
examples/NLG/
```
## Repository Overview

This repository includes multiple directories:
* The [data/](data) directory houses the raw datasets utilized in our experiments.
* The [src/](src) directory holds the source code for data manipulation, model training, and decoding.
* The [eval/](eval) directory includes scripts for evaluating performance on specific tasks.
* The [vocab/](vocab) directory contains the vocabulary files for GPT-2.

## Getting Started

Set up and install required packages within a virtual environment.:
 ```
 conda create -n NLG python=3.8.13
 conda activate NLG
 pip install -r requirement.txt
 bash download_pretrained_checkpoints.sh
 bash create_datasets.sh
 cd ./eval
 bash download_evalscript.sh
 cd ..
 ```

## Begin the E2E Task

1. Execute the training process for the GPT-2 Medium model employing MTLoRA. For the hyperparameters specific to GPT-2 Medium, please refer to our research paper
```
python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 8 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --save_interval 1000 \
    --lora_dim 4 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M/e2e \
    --random_seed 110
```

2. Create outputs using beam search with the trained model:
```
python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/e2e/model.26289.pt \
    --platform local \
    --lora_dim 4 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/e2e \
    --output_file predict.26289.b10p08r4.jsonl
```

3. Decode the results obtained in step (2)
```
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/e2e/predict.26289.b10p08r4.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref.txt \
    --output_pred_file e2e_pred.txt
```

4. Execute the evaluation process on the E2E test dataset

```
python eval/e2e/measure_scores.py e2e_ref.txt e2e_pred.txt -p
```

## Begin the WebNLG Task

1. Proceed with steps 1 and 2 of the E2E pipeline, substituting all mentions of E2E with webnlg. Refer to our paper for the specific hyperparameters

2. Decode the results generated from the beam search
```
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/webnlg/predict.20000.b10p08.jsonl \
    --input_file ./data/webnlg_challenge_2017/test_formatted.jsonl \
    --ref_type webnlg \
    --ref_num 6 \
    --output_ref_file eval/GenerationEval/data/references_webnlg \
    --output_pred_file eval/GenerationEval/data/hypothesis_webnlg \
    --tokenize --lower
```

3. Execute the evaluation process on the WebNLG test dataset
```
cd ./eval/GenerationEval/
python eval.py \
    -R data/references_webnlg/reference \
    -H data/hypothesis_webnlg \
    -nr 6 \
    -m bleu,meteor,ter 
cd ../..
```

## Begin the DART Task

1. Proceed with steps 1 and 2 of the E2E pipeline, substituting all mentions of E2E with dart. Refer to our paper for the specific hyperparameters

2. Decode the results generated from the beam search
```
python src/gpt2_decode.py \
        --vocab ./vocab \
        --sample_file ./trained_models/GPT2_M/dart/predict.20000.b10p08.jsonl \
        --input_file ./data/dart/test_formatted.jsonl \
        --ref_type dart \
        --ref_num 6 \
        --output_ref_file eval/GenerationEval/data/references_dart \
        --output_pred_file eval/GenerationEval/data/hypothesis_dart \
        --tokenize --lower
```

3. Execute the evaluation process on the Dart test dataset
```
cd ./eval/GenerationEval/
python eval.py \
    -R data/references_dart/reference \
    -H data/hypothesis_dart \
    -nr 6 \
    -m bleu,meteor,ter 
cd ../..
```

## Contact
Please contact us or post an issue if you have any questions.

* Yao Liang (liangyao2023@edwardjhu.com)
* Yuwei Wang (yuwei.wang@ia.ac.cn)
* Yi Zeng (yi.zeng@ia.ac.cn)

## Acknowledgements
This work was supported by the National Science and Technology Major Project (Grant No. 2022ZD0116202).

## Citation
```BibTeX
@misc{liang2024matrixtransformation,
      title={Matrix-Transformation Based Low-Rank Adaptation (MTLoRA): A Brain-Inspired Method for Parameter-Efficient Fine-Tuning}, 
      author={Yao Liang and Yuwei Wang and Yi Zeng},
      year={2024},
      eprint={2403.07440},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
