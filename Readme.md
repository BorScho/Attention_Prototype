# Attention Prototype for Tabular Data

## Goal

This project is a learning and exploration prototype for Transformer-based models on tabular business data.

The objective is not to build a production-ready account prediction model, but to understand the core concepts behind modern Transformer architectures and their application to tabular datasets. The project serves as a preparation step for studying recent tabular foundation models such as FT-Transformer, TabPFN, TabDPT and SAP TabICL.

## Dataset

The experiments use a subset of SAP FI accounting data derived from a HKONT (general ledger account) prediction use case.

For simplicity, only a small set of numerical and categorical features is used:

* WRBTR (amount)
* MWSKZ (tax code)
* WAERS (currency)
* BUDAT (posting date components)
* BLDAT (document date components)

Text fields and additional business features are intentionally excluded in the first prototype.

The target variable is HKONT.

For the initial experiments, the dataset is restricted to the 20 most frequent HKONT classes.

## Model

A minimal Transformer classifier was implemented in PyTorch from scratch:

* Feature embeddings for categorical variables
* Linear projection for numerical variables
* Learnable CLS token
* Learnable feature-position embeddings
* Transformer encoder stack
* Classification head

The implementation is intentionally simple and closely follows the concepts introduced in the original Transformer architecture.

## Results

The prototype demonstrates that a Transformer encoder can learn meaningful patterns from tabular accounting data.

Experimental results:

* Majority-class baseline: ~0.10 accuracy
* Transformer prototype: ~0.55 accuracy
* HistGradientBoosting baseline: ~0.63 accuracy

The Transformer clearly outperforms the majority baseline but remains below a strong tree-based baseline.

## Main Findings

The project highlights several important observations:

1. Transformer architectures can be applied to tabular business data.
2. Input representation is often more important than model depth.
3. Tree-based models remain very competitive on structured tabular datasets.
4. Building a Transformer from scratch provides valuable intuition for understanding modern tabular foundation models.

## Future Work

Possible extensions include:

* Text features (SGTXT, BKTXT)
* Improved feature tokenization
* FT-Transformer architecture
* Pretraining approaches
* Comparison with TabPFN and TabICL concepts

The current prototype should be viewed primarily as a study and experimentation platform rather than a production model.
