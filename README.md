# Bangla Agriculture Chatbot

A comparative research project for **Bangla agricultural question answering** using three instruction-tuned large language models—**Llama, Mistral, and Qwen**—evaluated across four major experimental stages:

1. **Baseline prompting**
2. **Retrieval-Augmented Generation (RAG)**
3. **Fine-Tuning**
4. **Fine-Tuning + RAG**

The project studies how prompting, retrieval grounding, parameter-efficient fine-tuning, and their combination affect answer quality for Bangla agricultural questions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Research Objective](#research-objective)
- [Models Used](#models-used)
- [Dataset](#dataset)
- [Experimental Pipeline](#experimental-pipeline)
- [Baseline Prompting Strategies](#baseline-prompting-strategies)
- [RAG Architecture](#rag-architecture)
- [Fine-Tuning Configuration](#fine-tuning-configuration)
- [Fine-Tuning + RAG](#fine-tuning--rag)
- [Repository Structure](#repository-structure)
- [Evaluation Metrics](#evaluation-metrics)
- [Complete Results](#complete-results)
- [Best Configuration per Model](#best-configuration-per-model)
- [Overall Findings](#overall-findings)
- [Important Experimental Notes](#important-experimental-notes)
- [How to Run](#how-to-run)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [License](#license)
- [Repository](#repository)

---

## Project Overview

This repository presents an experimental framework for evaluating LLM-based Bangla agricultural question answering.

Three LLM families are compared:

- **Llama**
- **Mistral**
- **Qwen**

Each model is evaluated under the same high-level workflow:

```text
Baseline Prompting
        ↓
       RAG
        ↓
   Fine-Tuning
        ↓
 Fine-Tuning + RAG
```

The project also compares multiple prompting strategies inside the baseline stage.

The central goal is to determine whether retrieval grounding and parameter-efficient fine-tuning can improve factual and semantic answer quality beyond prompting-only baselines.

---

## Research Objective

The main objectives are:

- Evaluate multiple LLM families on Bangla agricultural QA.
- Compare different baseline prompting strategies.
- Measure the impact of retrieval grounding.
- Measure the impact of fine-tuning.
- Investigate whether combining fine-tuning with RAG provides complementary gains.
- Compare lexical and semantic evaluation metrics.
- Identify the strongest model-stage combination.

---

## Models Used

The repository uses the following Hugging Face checkpoints:

| Model Family | Checkpoint |
|---|---|
| Llama | `meta-llama/Llama-3.2-1B-Instruct` |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` |
| Qwen | `Qwen/Qwen2.5-7B-Instruct` |

> **Note:** The models are not parameter-matched. Llama is a 1B model, while Mistral and Qwen are 7B models. Performance comparisons should therefore be interpreted as comparisons of the evaluated configurations rather than controlled parameter-scale comparisons.

---

## Dataset

The project contains three core datasets.

| File | Purpose | Size |
|---|---|---:|
| `Bangla_Agriculture_QA_Train_800.json` | Training data for fine-tuning | 800 QA pairs |
| `Bangla_Agriculture_QA_Test_200.json` | Evaluation set | 200 QA pairs |
| `Bangla_Agriculture_Knowledge_Base_500.json` | External knowledge base for RAG | 500 documents |

### Training/Test QA format

Typical fields include:

```json
{
  "id": "...",
  "question": "...",
  "reference_answer": "..."
}
```

### Knowledge-base format

Typical fields include:

```json
{
  "id": "...",
  "title": "...",
  "content": "...",
  "source_site": "...",
  "source_title": "...",
  "source_url": "..."
}
```

The data organization separates:

- supervised QA data for fine-tuning,
- a held-out QA set for evaluation,
- and an external knowledge base for retrieval.

---

## Experimental Pipeline

For each model, the project follows four main stages.

### Stage 1 — Baseline

The raw instruction-tuned model is evaluated without retrieval or task-specific fine-tuning.

### Stage 2 — RAG

The base model is augmented with external retrieved agricultural knowledge.

### Stage 3 — Fine-Tuning

The base model is adapted to the Bangla agricultural QA task using parameter-efficient fine-tuning.

### Stage 4 — Fine-Tuning + RAG

The fine-tuned model is combined with the same retrieval pipeline.

This provides a direct comparison of:

```text
Prompting-only
vs
Retrieval grounding
vs
Task adaptation
vs
Task adaptation + retrieval grounding
```

---

## Baseline Prompting Strategies

Each model is evaluated using five prompting strategies.

### 1. Zero-Shot

The model receives only the user question and answering instructions.

### 2. One-Shot

One example QA pair is included before the target question.

### 3. Few-Shot

Multiple demonstrations are included in the prompt.

### 4. Chain-of-Thought

The prompt encourages step-by-step reasoning before producing the answer.

### 5. Instruction Prompting

A more explicit task-oriented instruction is used to guide answer generation.

The baseline results are stored both as individual experiment outputs and as consolidated baseline summaries.

---

## RAG Architecture

The RAG implementation uses a **hybrid retrieval pipeline** rather than only dense vector search.

### Retrieval pipeline

```text
Knowledge Base
      ↓
Text Chunking
      ↓
Dense Retrieval ─────┐
                     ├──→ Reciprocal Rank Fusion
BM25 Retrieval ──────┘
                         ↓
                      Reranker
                         ↓
                 Top Retrieved Contexts
                         ↓
                       LLM
                         ↓
                    Final Answer
```

### Main retrieval components

| Component | Configuration |
|---|---|
| Dense embedding model | `BAAI/bge-m3` |
| Dense index | FAISS |
| Sparse retrieval | BM25 |
| Fusion | Reciprocal Rank Fusion |
| Initial candidate count | `25` |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Final retrieved contexts | Top 3 |
| Chunk size | 180 words |
| Chunk overlap | 30 words |
| Generation limit | approximately 320 new tokens |

This hybrid design allows the system to benefit from both:

- semantic similarity from dense embeddings,
- and exact lexical matching from BM25.

The reranker is used to improve final context selection before generation.

---

## Fine-Tuning Configuration

Fine-tuning is performed using a **QLoRA-style parameter-efficient setup with Unsloth**.

### Main configuration

| Parameter | Setting |
|---|---|
| Fine-tuning method | QLoRA / PEFT |
| Quantization | 4-bit |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Epochs | 5 |
| Learning rate | `2e-4` |
| Batch size | 2 |
| Gradient accumulation | 4 |
| Maximum sequence length | 1024 |
| Seed | 3407 |
| Packing | Enabled |

### Target modules

LoRA is applied to the main attention and feed-forward projection layers:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

This enables model adaptation without updating every model parameter.

---

## Fine-Tuning + RAG

The final stage combines:

```text
QLoRA-adapted model
        +
Hybrid RAG pipeline
```

The retrieval pipeline remains based on:

- BGE-M3 embeddings,
- FAISS,
- BM25,
- Reciprocal Rank Fusion,
- BGE reranking,
- Top-3 retrieved contexts.

This stage produces the strongest overall results in the repository.

---

## Repository Structure

```text
Bangla-Agriculture-Chatbot/
│
├── Data/
│   ├── Bangla_Agriculture_Knowledge_Base_500.json
│   ├── Bangla_Agriculture_QA_Test_200.json
│   └── Bangla_Agriculture_QA_Train_800.json
│
├── Llama/
│   ├── Baseline/
│   │   ├── Zero Shot/
│   │   ├── One Shot/
│   │   ├── Few Shot/
│   │   ├── Chain of Thought/
│   │   ├── Instruction/
│   │   ├── Baseline (Llama).ipynb
│   │   └── baseline_all_methods_result.csv
│   ├── RAG/
│   ├── Fine-Tune/
│   └── FineTune+RAG/
│
├── Mistral/
│   ├── Baseline/
│   │   ├── Zero Shot/
│   │   ├── One Shot/
│   │   ├── Few Shot/
│   │   ├── Chain of Thought/
│   │   ├── Instruction/
│   │   └── ...
│   ├── RAG/
│   ├── Fine-Tune/
│   └── FineTune+RAG/
│
├── Qwen/
│   ├── Baseline/
│   │   ├── Zero Shot/
│   │   ├── One Shot/
│   │   ├── Few Shot/
│   │   ├── Chain of Thought/
│   │   ├── Instruction/
│   │   └── ...
│   ├── RAG/
│   ├── Fine-Tune/
│   └── FineTune+RAG/
│
├── README.md
├── LICENSE
└── .gitignore
```

Each experiment directory generally contains one or more of:

- Jupyter notebooks,
- prediction CSV files,
- aggregate result CSV files,
- intermediate outputs.

---

## Evaluation Metrics

The project evaluates generated answers using both lexical and semantic metrics.

### Exact Match

Measures whether the generated answer exactly matches the reference answer.

### Fuzzy Match

Measures approximate string-level similarity.

### BLEU

Measures n-gram overlap between generated and reference answers.

### ROUGE-1

Measures unigram overlap.

### ROUGE-2

Measures bigram overlap.

### ROUGE-L

Measures longest-common-subsequence overlap.

### Token F1

Balances token-level precision and recall.

### BERTScore Precision

Measures semantic precision using contextual embeddings.

### BERTScore Recall

Measures semantic recall using contextual embeddings.

### BERTScore F1

Combines semantic precision and recall.

Because agricultural QA answers can be semantically correct without being exact lexical copies of the reference, BERTScore is especially useful alongside BLEU/ROUGE-style metrics.

---

## Complete Results

### Abbreviations

- **EM** = Exact Match
- **FM** = Fuzzy Match
- **R1** = ROUGE-1
- **R2** = ROUGE-2
- **RL** = ROUGE-L
- **TokF1** = Token F1
- **BP** = BERTScore Precision
- **BR** = BERTScore Recall
- **BF1** = BERTScore F1

| Model | Method | EM | FM | BLEU | R1 | R2 | RL | TokF1 | BP | BR | BF1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | Zero-shot | 0.000 | 0.4813 | 0.0289 | 0.1605 | 0.0560 | 0.1477 | 0.1605 | 0.7062 | 0.6982 | 0.7016 |
| Llama | One-shot | 0.000 | 0.4600 | 0.0187 | 0.1365 | 0.0420 | 0.1234 | 0.1365 | 0.6835 | 0.6805 | 0.6816 |
| Llama | Few-shot | 0.000 | 0.4010 | 0.0143 | 0.0698 | 0.0231 | 0.0639 | 0.0698 | 0.6459 | 0.6558 | 0.6504 |
| Llama | CoT | 0.000 | 0.4861 | 0.0271 | 0.1581 | 0.0532 | 0.1439 | 0.1581 | 0.7000 | 0.6976 | 0.6983 |
| Llama | Instruction | 0.000 | 0.4964 | 0.0281 | 0.1708 | 0.0582 | 0.1536 | 0.1708 | 0.7080 | 0.7029 | 0.7049 |
| Llama | RAG | 0.010 | **0.7016** | 0.2246 | 0.3898 | 0.2836 | 0.3612 | 0.3898 | 0.7532 | 0.7902 | 0.7698 |
| Llama | Fine-Tune | 0.000 | 0.4900 | 0.0291 | 0.1686 | 0.0537 | 0.1520 | 0.1686 | 0.7279 | 0.7266 | 0.7266 |
| **Llama** | **Fine-Tune + RAG** | **0.115** | 0.6925 | **0.2435** | **0.4457** | **0.3303** | **0.4211** | **0.4457** | **0.8172** | **0.7924** | **0.8037** |
| Mistral | Zero-shot | 0.000 | 0.4839 | 0.0201 | 0.1538 | 0.0436 | 0.1363 | 0.1538 | 0.6909 | 0.6999 | 0.6950 |
| Mistral | One-shot | 0.000 | 0.4948 | 0.0291 | 0.1698 | 0.0511 | 0.1525 | 0.1698 | 0.7038 | 0.7078 | 0.7053 |
| Mistral | Few-shot | 0.000 | 0.4643 | 0.0248 | 0.1369 | 0.0465 | 0.1253 | 0.1369 | 0.6673 | 0.6814 | 0.6739 |
| Mistral | CoT | 0.000 | 0.4808 | 0.0221 | 0.1559 | 0.0433 | 0.1391 | 0.1559 | 0.6928 | 0.7024 | 0.6972 |
| Mistral | Instruction | 0.000 | 0.4927 | 0.0219 | 0.1647 | 0.0413 | 0.1446 | 0.1647 | 0.7069 | 0.7050 | 0.7054 |
| Mistral | RAG | 0.000 | 0.7898 | 0.2803 | 0.4951 | 0.4096 | 0.4709 | 0.4951 | 0.7650 | 0.8296 | 0.7942 |
| Mistral | Fine-Tune | 0.000 | 0.5293 | 0.0384 | 0.2179 | 0.0721 | 0.1987 | 0.2179 | 0.7504 | 0.7505 | 0.7495 |
| **Mistral** | **Fine-Tune + RAG** | **0.615** | **0.8711** | **0.6727** | **0.7504** | **0.7238** | **0.7435** | **0.7504** | **0.9194** | **0.9023** | **0.9100** |
| Qwen | Zero-shot | 0.000 | 0.4993 | 0.0344 | 0.1781 | 0.0590 | 0.1599 | 0.1781 | 0.6974 | 0.7127 | 0.7045 |
| Qwen | One-shot | 0.000 | 0.4966 | 0.0329 | 0.1701 | 0.0552 | 0.1562 | 0.1701 | 0.6914 | 0.7068 | 0.6986 |
| Qwen | Few-shot | 0.000 | 0.5030 | 0.0335 | 0.1779 | 0.0593 | 0.1625 | 0.1779 | 0.6965 | 0.7129 | 0.7042 |
| Qwen | CoT | 0.000 | 0.5041 | 0.0368 | 0.1866 | 0.0613 | 0.1666 | 0.1866 | 0.7149 | 0.7227 | 0.7182 |
| Qwen | Instruction | 0.000 | 0.4504 | 0.0185 | 0.1498 | 0.0376 | 0.1375 | 0.1498 | 0.7111 | 0.6810 | 0.6946 |
| Qwen | RAG | 0.235 | 0.8818 | 0.4053 | 0.6464 | 0.5816 | 0.6270 | 0.6464 | 0.8450 | 0.8824 | 0.8611 |
| Qwen | Fine-Tune | 0.000 | 0.5148 | 0.0359 | 0.2044 | 0.0660 | 0.1884 | 0.2044 | 0.7441 | 0.7425 | 0.7424 |
| **Qwen** | **Fine-Tune + RAG** | **0.430** | **0.8891** | **0.6218** | **0.7540** | **0.7077** | **0.7440** | **0.7540** | **0.9227** | **0.9020** | **0.9112** |

---

## Best Configuration per Model

| Model | Best Configuration | BERTScore F1 |
|---|---|---:|
| Llama | Fine-Tune + RAG | **0.8037** |
| Mistral | Fine-Tune + RAG | **0.9100** |
| Qwen | Fine-Tune + RAG | **0.9112** |

### Best baseline configuration

| Model | Best Baseline | BERTScore F1 |
|---|---|---:|
| Llama | Instruction | **0.7049** |
| Mistral | Instruction | **0.7054** |
| Qwen | Chain-of-Thought | **0.7182** |

For Mistral, one-shot prompting performs slightly better on some lexical metrics, while instruction prompting gives the highest baseline BERTScore F1.

---

## Overall Findings

### 1. Fine-Tuning + RAG performs best

All three model families achieve their strongest overall results when fine-tuning and retrieval are combined.

### 2. Retrieval provides a major improvement

RAG provides much larger performance gains than fine-tuning alone.

For example, Qwen BERTScore F1 progresses as follows:

```text
Best Baseline:      0.7182
Fine-Tuning:        0.7424
RAG:                0.8611
Fine-Tuning + RAG:  0.9112
```

This suggests that access to relevant external knowledge is particularly important for the agricultural QA task.

### 3. Qwen achieves the strongest semantic result

The best BERTScore F1 is:

```text
Qwen + Fine-Tuning + RAG
BERTScore F1 = 0.9112
```

### 4. Mistral is strongest on several exact/lexical metrics

Mistral Fine-Tune + RAG achieves:

```text
Exact Match = 0.615
BLEU        = 0.6727
ROUGE-2     = 0.7238
```

Therefore, no single model dominates every evaluation metric.

### 5. Hybrid adaptation is complementary

Fine-tuning improves model behavior and task adaptation, while RAG provides external factual grounding.

Their combination produces the best overall results.

---

## Important Experimental Notes

### Different model sizes

The experiment compares:

```text
Llama 3.2 1B
Mistral 7B
Qwen 2.5 7B
```

Therefore, the results should not be interpreted as a pure architecture comparison.

### Generation-length differences

Some experiments use different maximum generation lengths.

Baseline generation is substantially shorter than later stages.

This leads to a higher number of truncated outputs in some baseline experiments.

Examples:

| Model | Baseline Method | Truncated Outputs |
|---|---|---:|
| Llama | Zero-shot | 112 / 200 |
| Llama | Few-shot | 191 / 200 |
| Mistral | Zero-shot | 158 / 200 |
| Qwen | Zero-shot | 151 / 200 |
| Qwen | Instruction | 49 / 200 |

Fine-tuned and hybrid systems show far fewer truncations.

This should be considered when interpreting stage-wise comparisons.

---

## How to Run

The experiments are organized as Jupyter notebooks and can be executed in environments such as:

- Kaggle
- Google Colab
- local GPU-enabled Jupyter
- compatible CUDA environments

### 1. Clone the repository

```bash
git clone https://github.com/RamijWasithRahat/Bangla-Agriculture-Chatbot.git
cd Bangla-Agriculture-Chatbot
```

### 2. Install dependencies

The exact dependency set varies slightly by notebook, but the project uses libraries such as:

```bash
pip install transformers datasets accelerate peft trl bitsandbytes
pip install sentence-transformers faiss-cpu rank-bm25
pip install evaluate rouge-score bert-score sacrebleu
pip install unsloth
```

For GPU environments, install versions compatible with the available CUDA/PyTorch configuration.

### 3. Prepare model access

Some Hugging Face models may require:

- a Hugging Face account,
- model-license acceptance,
- and an access token.

### 4. Run baseline experiments

Navigate to the baseline notebook under the selected model directory and execute the prompting strategies.

### 5. Run RAG

Use the model-specific RAG notebook.

The notebook loads the 500-document agricultural knowledge base, builds the retrieval pipeline, and evaluates the model on the 200-question test set.

### 6. Run fine-tuning

Run the model-specific fine-tuning notebook using the 800 training QA pairs.

### 7. Run Fine-Tuning + RAG

Load the adapted model and combine it with the hybrid retrieval pipeline.

### 8. Evaluate

Prediction files and aggregate result CSVs are generated under the corresponding experiment directories.

---

## Reproducibility

The repository retains:

- model-specific experiment notebooks,
- predictions for individual test questions,
- aggregate metric files,
- shared training/test datasets,
- shared RAG knowledge base,
- and fixed fine-tuning settings.

The fine-tuning seed used in the experiments is:

```text
3407
```

For a stricter controlled comparison, future reruns should use identical:

- generation limits,
- decoding settings,
- random seeds,
- hardware,
- package versions,
- and evaluation code.

---

## Limitations

The current experimental design has several limitations.

1. **Model size is not controlled.**  
   Llama uses a 1B model while Qwen and Mistral use 7B models.

2. **Generation budgets differ between stages.**  
   Baseline methods use shorter output limits than several later experiments.

3. **The dataset is relatively small.**  
   The study contains 800 training QA pairs, 200 test questions, and 500 knowledge-base documents.

4. **Automatic metrics do not fully measure factual correctness.**  
   Human evaluation or factuality-focused evaluation could strengthen the analysis.

5. **The current domain is limited to Bangla agriculture.**  
   Generalization to other languages and domains has not been established.

6. **Efficiency is not yet a central benchmark.**  
   Latency, memory usage, retrieval cost, throughput, and deployment cost could be evaluated more systematically.

---

## Future Work

Potential extensions include:

- human expert evaluation,
- hallucination analysis,
- factual consistency evaluation,
- retrieval ablation studies,
- retrieval recall/precision analysis,
- reranker ablation,
- chunk-size sensitivity experiments,
- Top-k sensitivity analysis,
- multilingual agricultural QA,
- larger Bangla agricultural datasets,
- model-size-controlled comparisons,
- latency and memory benchmarking,
- quantized deployment experiments,
- uncertainty-aware answer generation,
- citation-supported answers,
- conversation-aware agricultural assistants,
- and practical deployment for farmers and agricultural service providers.

---

## License

This repository is distributed under the license included in the project:

```text
MIT License
```

See the [`LICENSE`](LICENSE) file for details.

---

## Repository

**GitHub:**  
https://github.com/RamijWasithRahat/Bangla-Agriculture-Chatbot

---

## Summary

This project provides a systematic comparison of three LLM families for Bangla agricultural question answering across prompting, RAG, fine-tuning, and hybrid RAG + fine-tuning settings.

The main experimental result is:

```text
Best semantic performance:
Qwen2.5-7B-Instruct + QLoRA + Hybrid RAG
BERTScore F1 = 0.9112

Best Exact Match:
Mistral-7B-Instruct-v0.3 + QLoRA + Hybrid RAG
Exact Match = 0.615
```

Overall, the results indicate that **retrieval grounding provides substantial gains over prompting-only and fine-tuning-only approaches, while Fine-Tuning + RAG provides the strongest overall performance across all three model families.**
