# FINAL METRICS

**Dataset:** TwiBot-22

---

## FEATURE (Random Forest – Metadata Branch)

### Balanced vs Unweighted

| Configuration            | Accuracy | Precision | Recall | F1-Score | MCC    |
|--------------------------|----------|-----------|--------|----------|--------|
| Default (Unweighted)     | 0.7604   | 0.7754    | 0.2622 | 0.3918   | 0.3509 |
| Cost-Sensitive (Balanced)| 0.7024   | 0.4957    | 0.6160 | 0.5493   | 0.3354 |

### Feature Subset Ablation

| Feature Subset              | Count | Accuracy | Precision | Recall | F1-Score | MCC    |
|-----------------------------|-------|----------|-----------|--------|----------|--------|
| All Features (Baseline)     | 24    | 0.7016   | 0.4947    | 0.6176 | 0.5493   | 0.3349 |
| Top-15 Gini Features        | 15    | 0.7077   | 0.5031    | 0.6004 | 0.5475   | 0.3370 |
| Top-10 Gini Features        | 10    | 0.7087   | 0.5045    | 0.5928 | 0.5451   | 0.3352 |
| Without Textual Metrics     | 17    | 0.6832   | 0.4700    | 0.5959 | 0.5255   | 0.2973 |
| Only Textual Metrics        | 7     | 0.7097   | 0.5065    | 0.5436 | 0.5244   | 0.3162 |
| Only Log-Transformed Counts | 4     | 0.6751   | 0.4583    | 0.5677 | 0.5071   | 0.2724 |
| Only Behavioral Ratios      | 4     | 0.6891   | 0.4729    | 0.4879 | 0.4803   | 0.2587 |
| Only Boolean Flags           | 9     | 0.6279   | 0.4106    | 0.6052 | 0.4892   | 0.2231 |

---

## TEXT (RoBERTa – Content Branch)

### Timeline Sample Size (N tweets)

| N  | Accuracy | Precision | Recall | F1-Score | MCC   |
|----|----------|-----------|--------|----------|-------|
| 10 | 0.719    | 0.622     | 0.121  | 0.202    | 0.177 |
| 20 | 0.720    | 0.619     | 0.131  | 0.216    | 0.183 |
| 30 | 0.720    | 0.622     | 0.127  | 0.211    | 0.182 |

### Class Imbalance Strategies (N=20, mean ± std over 10 runs)

| Strategy      | Accuracy        | Precision       | Recall          | F1-Score        | MCC             |
|---------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| Base Model    | 0.720 ± 0.002   | 0.619 ± 0.018   | 0.131 ± 0.013   | 0.216 ± 0.016   | 0.183 ± 0.007   |
| Class Weights | 0.670 ± 0.005   | 0.462 ± 0.005   | 0.722 ± 0.008   | 0.563 ± 0.001   | 0.339 ± 0.003   |
| Oversampling  | 0.672 ± 0.008   | 0.464 ± 0.007   | 0.723 ± 0.013   | 0.565 ± 0.002   | 0.342 ± 0.004   |

---

## TEMPORAL (BiLSTM – Temporal Branch)

| Strategy       | Accuracy | Precision | Recall | F1-Score | MCC   |
|----------------|----------|-----------|--------|----------|-------|
| Base           | 0.706    | 0.000     | 0.000  | 0.000    | 0.000 |
| Class Weighted | 0.540    | 0.370     | 0.795  | 0.504    | 0.217 |

---

## ENSEMBLE (Meta-Classifier Stacking)

Features: prob_rf, prob_roberta (top-15 RF features, oversampled RoBERTa)

### Logistic Regression Meta-Classifier

| Configuration         | Accuracy | Precision | Recall | F1-Score | MCC    |
|-----------------------|----------|-----------|--------|----------|--------|
| Balanced class weights| 0.7047   | 0.4989    | 0.6956 | 0.5811   | 0.3744 |
| No class weights      | 0.7755   | 0.6722    | 0.4633 | 0.5486   | 0.4182 |

### Random Forest Meta-Classifier

| Configuration         | Accuracy | Precision | Recall | F1-Score | MCC    |
|-----------------------|----------|-----------|--------|----------|--------|
| Balanced class weights| 0.7349   | 0.5720    | 0.3952 | 0.4674   | 0.3077 |

