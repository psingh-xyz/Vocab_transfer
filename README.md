
# Vocabulary Transfer for Biomedical Texts: Add Tokens if You Cannot Add Data

## Overview

This work addresses the critical challenge of limited data in medical Natural Language Processing (NLP) by introducing a novel vocabulary transfer technique. The study focuses on expanding and adapting tokenization strategies for biomedical texts, demonstrating how vocabulary extension can significantly improve model performance in domains with scarce data.

## Key Features

- Domain-specific vocabulary expansion
- Innovative tokenization adaptation for medical texts
- Masked Language Modeling (MLM) intermediate step
- Applicable to low-resource NLP domains
- Improvements in both model accuracy and inference time

## Key Findings

- Increasing vocabulary size leads to measurable performance improvements
- Intermediate MLM step crucial for model adaptation
- Potential to reduce inference time in medical NLP tasks
- Demonstrated effectiveness on medical text classification datasets
- Promising approach for other domain-specific NLP challenges

## Methodology

Our approach involves the following steps:

1. Domain-specific vocabulary expansion
2. Tokenization adaptation for medical texts
3. Masked Language Modeling (MLM) intermediate step
4. Fine-tuning on target biomedical datasets

## Datasets Used

We employed two primary datasets for our research:

1. OHSUMED: Medical dataset for cardiovascular disease classification
   - Contains abstracts of PubMed documents from 23 cardiovascular disease classes
   - Used for evaluating the effectiveness of our vocabulary transfer technique

2. Kaggle Medical Texts Dataset: Classification of various patient conditions
   - Includes a wide range of medical conditions beyond cardiovascular diseases
   - Utilized to test the generalizability of our approach across different medical domains
  
## Paper

Published at: [https://arxiv.org/html/2208.02554v3](https://arxiv.org/html/2208.02554v3)

