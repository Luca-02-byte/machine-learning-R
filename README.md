# Machine Learning in R via tidyverse and tidymodels

![R](https://img.shields.io/badge/R-4.6+-276DC3?style=flat&logo=r&logoColor=white)
![Tidymodels](https://img.shields.io/badge/Framework-Tidymodels-blue?style=flat)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat)

Progetto strutturato in **R** che implementa, confronta e valuta algoritmi di **apprendimento supervisionato e non supervisionato** su dataset reali. Il codice copre: esplorazione dei dati (EDA), split del dataset in *training/validation/test set*, tuning degli iperparametri e valutazione comparativa delle metriche.

> **Fonte:** Basato sui laboratori del corso *Machine Learning* ([MLFE 25-26 R Labs](https://mcameletti.quarto.pub/mlfe-25-26-rlabs/)).

---

## Overview

| Modulo | Dataset | Obiettivo | Modelli Implementati |
| :--- | :--- | :--- | :--- |
| **A. Regressione** | `Boston` (`MASS`) | Prevedere il valore mediano delle abitazioni (`medv`) | Lineare, Step Functions, Polinomiale, Spline (Cubiche & Naturali), KNN, Alberi decisionali, Bagging, Gradient Boosting |
| **B. Classificazione** | `penguins` (`modeldata`) | Classificare il sesso (`sex`) in base a specie e morfologia | Logistic Regression, LDA, QDA, Naive Bayes, KNN, Alberi con Pruning, Random Forest |
| **C. Non Supervisionato** | `penguins` | Individuare i gruppi naturali (specie) dalle misure fisiche | Clustering Gerarchico, K-Means, Principal Component Analysis |

---

## Scelte Metodologiche

* **Prevenzione del Data Leakage:** Gli iperparametri (es. grado del polinomio $d$, numero di vicini $K$) vengono ottimizzati esclusivamente sul **Validation Set**. Solo dopo il tuning, *training* e *validation* vengono riuniti per addestrare il modello finale, valutato su un **Test Set** mai visto in precedenza.
* **Confronto Equo e Metriche:**
  * *Regressione:* Valutazione tramite **Test MSE** e correlazione osservati/previsti, confrontando l'uso del solo predittore principale (`lstat`) rispetto all'intero set di variabili.
  * *Classificazione:* Analisi di **Accuracy, Sensitivity, Specificity**, impatto dello spostamento della soglia decisionale ($0.5 \rightarrow 0.8$) e confronto tramite **Curve ROC e AUC**.
  * *Unsupervised:* Standardizzazione delle variabili ($Z$-score), scelta ottimale dei cluster (*Elbow Method*), biplot sulle prime componenti principali e calcolo della **Purezza (Purity)** rispetto alle specie reali.

---

## Struttura della Repository

```text
.
├── ML_project.R    # Script R completo, commentato e riproducibile
└── README.md       # Documentazione del progetto
