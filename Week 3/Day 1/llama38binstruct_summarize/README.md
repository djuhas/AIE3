---
license: other
library_name: peft
tags:
- trl
- sft
- generated_from_trainer
base_model: NousResearch/Meta-Llama-3-8B-Instruct
datasets:
- generator
model-index:
- name: llama38binstruct_summarize
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# llama38binstruct_summarize

This model is a fine-tuned version of [NousResearch/Meta-Llama-3-8B-Instruct](https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct) on the generator dataset.
It achieves the following results on the evaluation set:
- Loss: 1.8992

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: constant
- lr_scheduler_warmup_steps: 0.03
- training_steps: 100

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 1.4662        | 1.1364 | 25   | 1.3819          |
| 0.6972        | 2.2727 | 50   | 1.5203          |
| 0.3476        | 3.4091 | 75   | 1.8026          |
| 0.0995        | 4.5455 | 100  | 1.8992          |


### Framework versions

- PEFT 0.11.1
- Transformers 4.41.2
- Pytorch 2.3.1+cu121
- Datasets 2.20.0
- Tokenizers 0.19.1