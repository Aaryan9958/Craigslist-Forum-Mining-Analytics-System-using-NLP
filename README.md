# Craigslist Forum Mining Analytics

**Goal:** Turn unstructured feedback from Craigslist forums into structured, actionable product insights using an end‑to‑end NLP pipeline (scraping → preprocessing → sentiment → topics → clustering → supervised classification → dashboard assets).

---

## 📦 Repository Structure

```
craigslist-forum-analytics/
├─ data/                # (optional) sample data or exports
├─ models/              # trained models (.pkl) — large files are gitignored by default
├─ notebooks/           # Jupyter notebooks (final_total_code.ipynb)
├─ src/                 # optional: scripts if you modularize the pipeline
├─ docs/                # Milestone reports & presentation PDFs
├─ images/              # export charts/wordclouds here for README
├─ requirements.txt     # Python dependencies
├─ .gitignore
├─ LICENSE
└─ README.md
```

## 🧠 Project Overview

Craigslist hosts public forums (Feedback, Flag Help, Help Desk) with rich user comments. This project mines those discussions to surface **sentiment, recurring pain points, and themes** so product & moderation teams can prioritize fixes and improvements.

**Pipeline:**
1) **Web Scraping** — `requests`, `BeautifulSoup` to collect forum threads/posts  
2) **Preprocessing** — lowercase, de‑noise, stopword removal, lemmatization  
3) **Sentiment Analysis** — VADER baseline + *frustration keyword override* for better complaint detection  
4) **Topic Modeling** — LDA (gensim) to discover themes (e.g., posting/visibility, UX/navigation, spam/messaging, etc.)  
5) **Vectorization & Clustering** — TF‑IDF (1–2 grams) → TruncatedSVD (100 comps) → KMeans (k≈5) + silhouette  
6) **Supervised Models** — Logistic, SVM, RandomForest → **Hard Voting Ensemble** for final classification  
7) **Reporting** — confusion matrices, topic/cluster keywords, word clouds, and a summarized dashboard

---

## 🚀 Quickstart

1) **Create a virtual environment** and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon stopwords wordnet punkt
```

2) **Open the notebook**
```bash
jupyter notebook notebooks/final_total_code.ipynb
```
Run cells top‑to‑bottom. The notebook covers scraping (or loading saved data), preprocessing, modeling, and evaluation.

---

## 📊 Results (at a glance)

- **Topics**: recurring issues around post visibility/flagging, UX/navigation, formatting, spam/messaging, and mixed sentiment about simplicity
- **Classifier**: Voting Ensemble (Logit + SVM + RF) achieved solid performance (accuracy and macro‑F1 in the 70s range) with strong recall on negatives (helpful for complaint triage)
- **Artifacts**: confusion matrix, topic keyword tables, word clouds, explained‑variance plot.

---

## 🖼️ Visuals
[View Title Slide](images/slide_01.jpg) ·
[Pipeline](images/slide_02.jpg) ·
[Scraping & Preprocessing](images/slide_03.jpg) ·
[Sentiment](images/slide_04.jpg) ·
[Topics](images/slide_05.jpg)




---

## 📥 Data & Models

- Place sample/scraped data under `/data` (keep sensitive/raw data out of the repo).  
- Trained model files live in `/models`. Consider **Git LFS** if you need to version binary artifacts.

---

## 🧪 Repro Notes

- If scraping, throttle requests and respect robots.txt/ToS.  
- For consistent results, fix random seeds where possible and note library versions.  
- Track parameters (e.g., number of topics, k for KMeans) in notebook cells or a config file.

---

## 🛡️ License

MIT — see `LICENSE`.
