# Evaluating the Efficiency of Deep Networks in Hierarchical Text Classification

This repository contains the implementation of experiments based on the paper:  
**TELEClass: Taxonomy Enrichment and LLM-Enhanced Hierarchical Text Classification with Minimal Supervision**.  

We replicate the TELEClass framework and replace the original encoder with **BERT** and **DeBERTa** to evaluate their effectiveness in hierarchical text classification.

---

## 📌 Project Structure
- `data/` : Datasets used for training and evaluation  
- `models/` : Scripts for loading BERT and DeBERTa models  
- `training/` : Training scripts for hierarchical classification  
- `results/` : Experimental results (tables, logs, and figures)  
- `README.md` : Project description  

---

## 🚀 Methodology
1. **Taxonomy Enrichment** – Expanding class labels using LLMs and corpus-based keywords.  
2. **Core Class Refinement** – Assigning pseudo-labels through embedding similarity.  
3. **Classifier Training** – Training hierarchical classifiers with BERT and DeBERTa as encoders.  

---

## 📊 Results

| Model   | Precision@1 | Precision@2 | Precision@3 | MRR    | Example F1 |
|---------|-------------|-------------|-------------|--------|------------|
| BERT    | 0.2399      | 0.1647      | 0.1515      | 0.1714 | 0.1553     |
| DeBERTa | 0.2399      | 0.2474      | 0.2067      | 0.1906 | 0.2099     |
| Simple Classifier (from paper) | 0.8505 | --- | 0.6421 | 0.6865 | 0.6483 |

**Observation**: DeBERTa outperforms BERT across most metrics, but both models perform lower compared to the original simple classifier from the TELEClass paper.



# Hirichal-Text-classification
# Hirichal-Text-classification
# Hirichal-Text-classification
