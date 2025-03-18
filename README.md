# VulKey:Automated Vulnerability Repair Guided by Domain-Specific Repair Patterns

## Catalog
- CodeGen: Repair Code Generation Model
    - Code: train, test code [CodeLlama, DSCoder, QwenCoder, StarCoder, VulRepair]
    - Data: train, test data [Transfer, PrimeVul, Vul4J]
- KnowledgeMatch: Expert Knowledge Matching Model
    - PrimeVul
        - Code
        - Data
    - X1
        - Code
        - Data



## How to reproduce
- Train by `train.sh`
- Test by `test.sh`

# Note
- Download `https://huggingface.co/LLM4APR/CodeLlama-70B_for_NMT` and `https://huggingface.co/LLM4APR/StarCoder-15B_for_NMT` for FT-BUG