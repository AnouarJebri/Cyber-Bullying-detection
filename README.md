# Cyberbullying Detection – Text Classification with spaCy and RoBERTa

This project focuses on detecting cyberbullying, offensive language, and hate speech using a text classification model trained with spaCy and a RoBERTa-base transformer. The goal is to classify user comments into three categories:

* Normal
* Offensive
* Hateful

The final trained model is exported and hosted on Google Drive.

---

## 1. Project Objective

The objective is to build an automated moderation system capable of identifying harmful content. This project includes:

* A multi-class text classifier
* Training pipelines using spaCy v3
* Visualization of training loss and evaluation metrics
* Exporting a production-ready model

---

## 2. Dataset

The dataset contains short social media comments labeled as:

| Label     | Description                        |
| --------- | ---------------------------------- |
| NORMAL    | Safe, harmless comment             |
| OFFENSIVE | Rude or insulting content          |
| HATEFUL   | Content targeting protected groups |

The dataset is converted to spaCy’s binary `.spacy` format for efficient training.

---

## 3. Training Pipeline

The model is trained using:

* Backbone: roberta-base (via spacy-transformers)
* Components: `transformer`, `textcat_multilabel`
* Optimizer: AdamW
* Loss: Cross entropy for text classification
* Evaluation logged every 200 steps

Training stabilizes around step 4000–10000, with CATS_SCORE values between 61% and 64%.

---

## 4. Repository Structure

```
project/
 ├── config.cfg
 ├── training_data.spacy
 ├── training_log.csv
 ├── plot_training.ipynb
 ├── model-best/
 └── README.md
```

---

## 5. Model Download

Download the trained model from Google Drive:

**Model Link:**
[https://drive.google.com/drive/folders/1kcoCyIMFXuefo3biAlNCJs9MXolrPHaX?usp=sharing](MODEL-LINK)



---

## 6. Installation

Install dependencies:

```bash
pip install spacy==3.7.2 spacy-transformers pandas matplotlib
```

---

## 7. Using the Model

```python
import spacy

nlp = spacy.load("path/to/model-best")

doc = nlp("You are such a loser, nobody likes you.")

print(doc.cats)
```

Example output:

```json
{
  "NORMAL": 0.01,
  "OFFENSIVE": 0.88,
  "HATEFUL": 0.03
}
```

---

## 8. Training Metrics Visualization

Training logs are plotted using matplotlib:

```python
plt.plot(df["#"], df["LOSS_TRANS"], label="Transformer Loss")
plt.plot(df["#"], df["LOSS_TEXTCAT"], label="TextCat Loss")
plt.plot(df["#"], df["CATS_SCORE"], label="Cats Score")
plt.plot(df["#"], df["SCORE"], label="Overall Score")
plt.legend()
plt.show()
```

---

## 9. Deployment Options

The model can be deployed as:

* A FastAPI inference service
* A Dockerized microservice
* An API endpoint integrated into a moderation system

If you want, I can generate the full FastAPI server, Dockerfile, or a ready-to-use API.

---

## 10. Author

Anouar Jebri
Cyberbullying text classification project – 2025
