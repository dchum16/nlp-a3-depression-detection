# Mental Health Signal Detection in Social Media Text

A fine-tuned DistilBERT classifier for detecting depression-indicative language in Reddit posts. Developed for **NLP Assessment 3 (41043/42850 - Natural Language Processing Algorithms, UTS, Autumn 2026).**

---

## Overview

This project applies natural language processing (NLP) techniques to identify depression signals in user-generated social media text. A pre-trained DistilBERT model is fine-tuned on a publicly available Reddit dataset to perform binary classification (depression vs non-depression). The system is benchmarked against a TF-IDF + Logistic Regression baseline to demonstrate the advantage of contextual transformer-based representations over sparse lexical features.

The intended framing of this system is as an **early signal screening tool** for online mental health spaces — not a clinical diagnostic instrument.

---

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.9483 | 0.9742 | 0.9200 | 0.9463 |
| **DistilBERT (fine-tuned)** | **0.9802** | **0.9929** | **0.9670** | **0.9797** |

DistilBERT outperformed the baseline across all metrics, with particularly strong gains in recall on the depression class — a critical metric for screening applications.

---

## Repository Contents

| File | Description |
|---|---|
| `nlp_a3_depression_detection.ipynb` | Main Colab notebook with the full pipeline |
| `depression_dataset_reddit_cleaned.csv` | The dataset used for training and evaluation |
| `figures/` | All plots referenced in the report (class distribution, confusion matrices, attention heatmap, training curves) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## Notebook Structure

The notebook is organised into self-contained sections:

1. **Setup** — imports and GPU check
2. **Load Dataset** — Drive mount and data loading
3. **Exploratory Data Analysis** — class distribution, post lengths, sample posts
4. **Preprocessing** — train/val/test split and tokenisation
5. **Baseline** — TF-IDF + Logistic Regression
6. **Fine-tune DistilBERT** — model setup, training loop, weighted loss
7. **Evaluation** — classification reports, confusion matrices, comparison table
8. **Attention Visualisation** — interpretability via attention heatmaps
9. **Error Analysis** — qualitative inspection of misclassified posts

---

## How to Run

1. Open the notebook in Google Colab
2. Set the runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU)
3. Mount your Google Drive when prompted
4. Update the dataset path in Section 2 to point to your local copy
5. Run all cells (Runtime → Run all)

The full pipeline takes roughly 15–20 minutes on the free Colab T4 GPU.

---

## Dataset

**Source:** [Depression: Reddit Dataset (Cleaned)](https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned) — Kaggle

- 7,731 Reddit posts
- Binary labels (`is_depression`: 1 = depression-indicative, 0 = non-depression)
- Approximately balanced classes (49.6% / 50.4%)

Posts labelled as depression-indicative were sourced from the r/depression subreddit; non-depression posts were sourced from non-mental-health subreddits such as r/CasualConversation.

---

## Tech Stack

- Python 3.12
- PyTorch
- Hugging Face Transformers
- scikit-learn
- pandas, NumPy
- matplotlib, seaborn
- Google Colab (T4 GPU)

---

## Limitations and Ethical Considerations

This system is a **research prototype**, not a deployable diagnostic tool. Key limitations:

- Reddit users are not demographically representative of the general population
- Labels are inferred from subreddit membership, not verified clinical diagnoses
- Model may struggle with sarcasm, irony, and out-of-distribution language
- Single-platform, single-snapshot data limits generalisation to other platforms or time periods

Any real-world application of this kind of system should involve human oversight, transparent communication of limitations, and appropriate referral pathways to qualified mental health professionals.

---

## Group Members

| Name | Student ID |
|---|---|
| Ibrahim Kashif| [25261090] |
| Ayaan Khurram Saghir| [25556263] |
| Dylan Chum | [14623413] |
| Luca Martins | [Student ID] |



