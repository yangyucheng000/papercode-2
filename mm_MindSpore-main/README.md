# Prototype-guided Knowledge Transfer for Federated Unsupervised Cross-modal Hashing

This library contains a MindSpore implementation of 'Prototype-guided Knowledge Transfer for Federated Unsupervised Cross-modal Hashing'

## Datasets

* MIRFLICKR-25K consists of 25,000 image-tags pairs with 24 unique concepts. Following the experimental setting in [13, 41, 57], 20,015 image-tags pairs are selected to perform the cross-modal retrieval task. We randomly select 2,000 pairs as the query set and the remaining 18,015 pairs as the retrieval set. We select 5,000 pairs from the retrieval set as the training set.

## Requirements

* Python==3.7
* Mindspore==2.1.1

## Run

```powershell
python main.py 
```

