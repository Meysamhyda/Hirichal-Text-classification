# Evaluating the Efficiency of Deep Networks in Hierarchical Text Classification

This repository contains the implementation of experiments based on the paper:  
**TELEClass: Taxonomy Enrichment and LLM-Enhanced Hierarchical Text Classification with Minimal Supervision**.  

We replicate the TELEClass framework and replace the original encoder with **RoBERTa** and **DeBERTa** to evaluate their effectiveness in hierarchical text classification.

---

---

## 🚀 Methodology
1. **Taxonomy Enrichment** – Expanding class labels using LLMs and corpus-based keywords.  
2. **Core Class Refinement** – Assigning pseudo-labels through embedding similarity.  
3. **Classifier Training** – Training hierarchical classifiers with BERT and DeBERTa as encoders.  

---

## 📊 Results

| Model   | Precision@1 | Precision@2 | Precision@3 | MRR    | Example F1 |
|---------|-------------|-------------|-------------|--------|------------|
| RoBERTa  | 0.2399      | 0.1647      | 0.1515      | 0.1714 | 0.1553     |
| DeBERTa | 0.2399      | 0.2474      | 0.2067      | 0.1906 | 0.2099     |


**Observation**: DeBERTa outperforms RoBERTa across most metrics, but both models perform lower compared to the original simple classifier from the TELEClass paper because training model on 2 epochs.



