# Geometry-Aware Optimal Transport for Interpretable Semantic Textual Similarity

Final year thesis (PFE) — INSEA, 3D Smart Factory.

## Overview

This thesis investigates whether replacing the Euclidean ground metric in Word Mover's Distance (WMD) with hyperbolic or spherical geometry alternatives improves semantic textual similarity on STS-B. The core hypothesis is that hyperbolic geometry — which naturally encodes hierarchical structure — should produce more meaningful word-level transport costs for sentence pairs where similarity is driven by hypernymy relations (e.g., "dog/animal", "car/vehicle"). The differentiating angle from transformer-based approaches (e.g., SBERT) is interpretability — geometry choice is an explicit, auditable design decision rather than a black-box learned representation. A word-level diagnostic on HyperLex confirms that Poincaré embeddings trained on WordNet capture hierarchical relations significantly better than Euclidean embeddings, justifying the geometric hypothesis before testing it at the sentence level.

## Key Finding
 
Performance depends on the **three-way alignment** between geometric space, training signal, and downstream task — not geometry alone. Hyperbolic geometry yields gains when all three align (Poincaré + WordNet + hypernymy evaluation), but produces null or negative results when the training signal (co-occurrence) or the task (paraphrase similarity) does not match the geometric prior.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `wmd_geometric_comparison.ipynb` | Three-way geometric WMD comparison (Euclidean / Spherical / Hyperbolic) on STS-B |
| `hyperbolic_evaluation.ipynb` | Word-level diagnostic — Poincaré embeddings on WordNet/HyperLex |
| `native_hyperbolic_wmd.ipynb` | Native Poincaré GloVe WMD on STS-B |
| `sbert_comparison.ipynb` | SBERT vs WMD — accuracy–interpretability trade-off |

> If GitHub's notebook preview fails (an intermittent issue with notebooks containing many inline figures), PDF exports are available in [`notebooks/exports/`](notebooks/exports/), or you can view the notebooks via [nbviewer](https://nbviewer.org/github/Saad1624/Geometry-Aware-Optimal-Transport-for-Interpretable-Semantic-Textual-Similarity/tree/main/notebooks/).

## Dependencies

### Python Environment
```bash
conda create -n poincare python=3.11
conda activate poincare
pip install gensim POT nltk datasets numpy scipy matplotlib seaborn sentence-transformers
```

### Poincaré GloVe
Clone Tifrea et al.'s repo and apply Python 3.11 compatibility fixes:
```bash
git clone https://github.com/alex-tifrea/poincare_glove.git
```

### Data
- STS-B benchmark — loaded via `datasets` library
- GloVe 300d — loaded via `gensim.downloader`
- text8 corpus — `wget http://mattmahoney.net/dc/text8.zip`

## References
- Kusner et al. (2015) — Word Mover's Distance
- Nickel & Kiela (2017) — Poincaré Embeddings
- Tifrea et al. (2019) — Poincaré GloVe
- Sala et al. (2018) — Representation Tradeoffs for Hyperbolic Embeddings
- Reimers & Gurevych (2019) — Sentence-BERT
- Cuturi (2013) — Sinkhorn Distances
