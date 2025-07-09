# AI Agent Fine-Tuning Project Plan

## Project Title
Fine-Tuning and Evaluating AI Agents for Customer Support Using Open-Source LLMs

## Objective
Build an AI agent using an open-source LLM and a publicly available dataset. Compare two different fine-tuning methods based on performance and practicality for business use cases, in line with Quantamind’s vision.

## Dataset

- Dataset Name: Twitter Customer Support Dataset
- Source: Kaggle (https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- Size: Original dataset is larger than 100MB; will downsample to ~100MB for fine-tuning
- Format: CSV containing columns such as `author_id`, `text`, `response`, `created_at`
- Subsetting Strategy:
  - Keep only top 5 brands (e.g., AppleSupport, XboxSupport, etc.)
  - Limit to a fixed number of cleaned, well-structured support conversation pairs

## Task Definition

- Task Type: Response Generation / Text2Text
- Input: Customer message
- Output: Agent reply
- Evaluation: Accuracy of response, similarity to ground-truth, coherence, and relevance

## Model

- Base LLM: Mistral-7B-Instruct (or smaller if compute-constrained)
- Model Source: Hugging Face
- Tokenizer: Corresponding tokenizer from model repo
- Quantization: 4-bit via bitsandbytes for efficiency
- Inference Pipeline: Transformers or vLLM

## Inference Logic / Engine

- Hybrid rule-based logic
- Rules include:
  - Escalation flags based on keywords (e.g., “cancel,” “angry,” “lawsuit”)
  - Priority score assigned based on intent
  - LLM uses prompt + scoring tag to formulate response
- Justification: Mimics enterprise logic layering on top of LLM behavior

## Fine-Tuning Method 1: QLoRA with PEFT

- Method: Parameter-Efficient Fine-Tuning using QLoRA
- Library: HuggingFace `transformers`, `peft`, `accelerate`
- Layers updated: Adapter layers only
- Precision: 4-bit quantization
- Training: Small GPU footprint, good performance
- Pros:
  - Lightweight
  - Fast to train
  - Modular and scalable
- Cons:
  - Slightly lower performance than full fine-tuning in some edge cases

## Fine-Tuning Method 2: Prompt Tuning

- Method: Lightweight soft prompt embeddings prepended to input
- Library: HuggingFace `peft`
- Layers updated: Prompt embeddings only
- Pros:
  - Extremely fast to train
  - Ideal for low-resource settings
  - Small storage footprint
- Cons:
  - May underperform on complex responses or longer context
  - Not always transferable between domains

## Metrics

- Main Metrics:
  - BLEU Score
  - ROUGE-L
  - Exact Match Rate
  - Latency (inference speed per prompt)
  - Model Size After Training
- Dataset Split:
  - Train: 80%
  - Validation: 10%
  - Test: 10%
- Qualitative: Manual review of generated responses for helpfulness, tone, and coherence


## Final Deliverables

- GitHub repo with:
  - Clean, modular codebase
  - Two separate fine-tuning methods implemented
  - Evaluation results and plots
  - Inference example script
- README includes:
  - Project overview
  - Dataset description
  - Model architecture and methods
  - Step-by-step instructions to reproduce
  - Performance metrics and comparison
  - Final conclusion and method recommendation

## Conclusion Goals

- Highlight trade-offs: performance vs efficiency
- Recommend the best method based on a realistic business use case
- Reflect on next steps (e.g., how to scale to 10x more data or production)

