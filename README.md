# 📰 News Article Classifier

A machine learning-powered web application that classifies news articles into **6 categories** using Natural Language Processing (NLP).

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red?logo=streamlit)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)

---

## 🎯 Features

- **Real-time Classification** — Paste any news article and get instant predictions
- **3 ML Models** — Naive Bayes, Logistic Regression, Linear SVM (compared side-by-side)
- **Confidence Scores** — See prediction probabilities for every category
- **Interactive Dashboard** — Confusion matrix, classification reports, accuracy charts
- **Word Clouds** — Visual representation of important words per category
- **Sample Articles** — Pre-loaded examples for quick demonstrations

## 📂 Categories

| Emoji | Category     | Description                    |
| ----- | ------------ | ------------------------------ |
| 💻    | Technology   | Computers, hardware, software  |
| 🏀    | Sports       | Baseball, hockey               |
| 🔬    | Science      | Space, medicine, electronics   |
| ⚡    | Politics     | Government, policy, debate     |
| 🏢    | Business     | Commerce, sales, trade         |
| 🚗    | Recreation   | Autos, motorcycles, hobbies    |

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **Dataset:** 20 Newsgroups (built into scikit-learn)
- **NLP:** TF-IDF Vectorization with bigrams
- **Models:** Multinomial Naive Bayes, Logistic Regression, Linear SVM
- **Web UI:** Streamlit
- **Visualization:** Plotly, Matplotlib, Seaborn, WordCloud

## 🚀 Quick Start

### 1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the models

```bash
python train_model.py
```

This will:
- Download the 20 Newsgroups dataset
- Train 3 ML models (Naive Bayes, Logistic Regression, SVM)
- Evaluate each model and print metrics
- Save trained models to the `models/` folder

### 4. Launch the web app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501` (or double-click `run_app.bat` on Windows).

### 5. CLI Demo (Optional)

Classify headlines directly from your command line:

```bash
python demo_cli.py "NASA rover discovered minerals on Mars"
```

Or start interactive mode:
```bash
python demo_cli.py
```

### 6. Run Tests

Verify pipeline and model inference:

```bash
python test_pipeline.py
```

## 📁 Project Structure

```
news_article_classifier/
├── .github/workflows/ci.yml # Automated CI test workflow
├── app.py                   # Streamlit web application
├── demo_cli.py              # CLI prediction interface
├── train_model.py           # Model training script
├── test_pipeline.py         # Unit & integration tests
├── requirements.txt         # Python dependencies
├── run_app.bat              # 1-click Windows web app launcher
├── run_cli.bat              # 1-click Windows CLI launcher
├── LICENSE                  # MIT License
├── README.md                # Project documentation
├── PRESENTATION_GUIDE.md    # Presentation & demo guide
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Dataset loading & category mapping
│   ├── preprocessor.py      # Text cleaning & TF-IDF
│   ├── model.py             # Model definitions & persistence
│   └── utils.py             # Metrics & helper functions
└── models/                  # Saved trained models (.pkl)
```

## 🧠 How It Works

```
Raw Text → Clean Text → TF-IDF Vectors → ML Model → Predicted Category
```

1. **Text Preprocessing:** Remove emails, URLs, special characters; lowercase; strip whitespace.
2. **TF-IDF Vectorization:** Convert cleaned text into numerical feature vectors using term frequency–inverse document frequency, with unigram + bigram features.
3. **Classification:** Three models are trained and compared — the user can switch between them in the app.

## 📊 Model Performance

| Model                | Accuracy  |
| -------------------- | --------- |
| Naive Bayes          | ~82%      |
| Logistic Regression  | ~87%      |
| Linear SVM           | ~87%      |

*(Actual numbers may vary slightly based on dataset split.)*

---

*Built with ❤️ for NLP and Machine Learning*
