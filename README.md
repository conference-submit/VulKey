# VulKey:Automated Vulnerability Repair Guided by Domain-Specific Repair Patterns

VulKey is a two-stage vulnerability repair framework consisting of expert knowledge matching and repair code generation. VulKey integrated the most advanced decoder-only LCMs as the code generation model while effectively incorporating domain knowledge as repair guidance through the three-level hierarchical abstraction of repair patterns.


## Project Structure

```
VulKey/
├── CodeGen/                    # Code Generation Models
│   ├── Code/                   # Model implementations
│   │   ├── CodeLlama/         # CodeLlama (FT-BUG, FT-VUL)
│   │   ├── DSCoder/           # DeepSeek Coder (FT-BUG, FT-VUL)  
│   │   ├── QwenCoder/         # Qwen Coder (FT-BUG, FT-VUL, KEY)
│   │   ├── StarCoder/         # StarCoder (FT-BUG, FT-VUL, KEY)
│   │   └── VulRepair/         # Baseline VulRepair model
│   └── Data/                  # Datasets
│       ├── Train_Datasets/    # PrimeVul, Transfer
│       └── Test_Benchmarks/   # Vul4J
└── KnowledgeMatch/            # Expert Knowledge Matching
    ├── PrimeVul/              # Knowledge matching for PrimeVul
    └── X1/                    # Knowledge matching for Vul4J
```

## Prerequisites

- Python 3.12 or higher
- PyTorch 2.5.1 or higher
- CUDA-compatible GPU (recommended)
- Git LFS for large model files


## Reproduce Guideline

### Training

#### 1. Code Generation Models

Train vulnerability repair models:

```bash
# CodeLlama
cd CodeGen/Code/CodeLlama/FT-VUL
bash train.sh

# StarCoder
cd CodeGen/Code/StarCoder/FT-VUL
bash train.sh

# DSCoder
cd CodeGen/Code/DSCoder/FT-BUG
bash train_trl.sh
cd CodeGen/Code/DSCoder/FT-VUL
bash train_trl.sh

# QwenCoder
cd CodeGen/Code/QwenCoder/FT-BUG
bash train_trl.sh
cd CodeGen/Code/QwenCoder/FT-VUL
bash train_trl.sh

# VulRepair (Baseline)
cd CodeGen/Code/VulRepair
bash train.sh
```

#### 2. Knowledge Matching Models

Train expert knowledge matching models:

```bash
# PrimeVul knowledge matching
cd KnowledgeMatch/PrimeVul/Code
bash train.sh

# Vul4J knowledge matching
cd KnowledgeMatch/X1/Code
bash train.sh
```

### Inference

#### 1. (RQ1) Evaluate on PrimeVul Dataset

```bash
# LCMs
cd CodeGen/Code/[MODEL_NAME]/FT-VUL
bash test.sh

# VulKey
cd KnowledgeMatch/PrimeVul/Code
bash test.sh

cd CodeGen/Code/StarCoder/KEY
bash test_vllm.sh

```

#### 2. (RQ3) Evaluate on Vul4J Benchmark

```bash
# LCMs
cd CodeGen/Code/[MODEL_NAME]/FT-VUL
bash test_v4j.sh

# VulKey
cd KnowledgeMatch/X1/Code
bash test.sh

cd CodeGen/Code/StarCoder/KEY
bash test_v4j_vllm.sh
```

#### 3. (RQ2) Abalation Study

```bash
cd CodeGen/Code/StarCoder/KEY
bash test_vllm_k.sh    # without action
bash test_vllm_a.sh    # without keyword
```


## Note
- Download `https://huggingface.co/LLM4APR/CodeLlama-70B_for_NMT` and `https://huggingface.co/LLM4APR/StarCoder-15B_for_NMT` for FT-BUG