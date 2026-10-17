# 🎓 News Article Classifier — Teacher Presentation Guide & Viva Cheat Sheet

This guide is designed to help you ace your project presentation and viva questions in front of your teacher.

---

## ⏱️ Quick Pitch (30-Second Elevator Pitch)

> *"Good morning/afternoon! Today I'm presenting an end-to-end Machine Learning and Natural Language Processing project: an automated **News Article Classifier**. Given raw, unformatted news text, our system cleans the data, transforms it into numerical feature vectors using TF-IDF with n-grams, and classifies it into one of six core categories — Technology, Sports, Science, Politics, Business, or Recreation. We benchmarked three distinct algorithms — Naive Bayes, Logistic Regression, and Linear SVM — achieving over **85.6% accuracy**, and deployed the best model into an interactive Streamlit dashboard with real-time confidence scores and diagnostic visualizations."*

---

## 📋 5-Minute Presentation Structure

### 1. Slide / Topic 1: The Problem & Objective
- **Problem:** Thousands of news articles are published every minute. Manual tagging is slow, error-prone, and expensive.
- **Objective:** Build an intelligent, automated NLP pipeline to categorize text accurately and explain its predictions with confidence scores.

### 2. Slide / Topic 2: Dataset & Preprocessing
- **Dataset:** The 20 Newsgroups corpus (scikit-learn benchmark) mapped into **6 clean categories**:
  - 💻 Technology
  - 🏀 Sports
  - 🔬 Science
  - ⚡ Politics
  - 🏢 Business
  - 🚗 Recreation
- **Cleaning Pipeline:**
  1. Lowercasing
  2. Regex-based stripping of email addresses and URLs
  3. Removal of punctuation, numbers, and special symbols
  4. Extra whitespace compression

### 3. Slide / Topic 3: Feature Engineering (TF-IDF)
- **Why TF-IDF?**
  - **Term Frequency (TF):** Measures how frequently a word appears in an article.
  - **Inverse Document Frequency (IDF):** Penalizes words that appear everywhere across all articles (like common conversational filler).
  - Formula: $\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)$
- **Key Enhancements:**
  - **Unigrams & Bigrams (1, 2):** Captures phrases like *"machine learning"* or *"space telescope"*, not just isolated words.
  - **Sublinear TF scaling:** Uses $1 + \log(\text{TF})$ to dampen the dominance of repetitive words.
  - **English Stopwords Removal:** Filters out trivial words (`the`, `is`, `at`).

### 4. Slide / Topic 4: Model Comparison & Benchmarks

| Model | Architecture | Accuracy | Training Time | Key Strength |
| :--- | :--- | :---: | :---: | :--- |
| **Naive Bayes** | MultinomialNB ($\alpha=0.1$) | **85.21%** | 4.6s | Extremely fast, strong probabilistic baseline |
| **Logistic Regression** | L2 Regularized, L-BFGS | **85.02%** | 7.0s | Well-calibrated multi-class probabilities |
| **🥇 Linear SVM** | SGDClassifier + Platt Calibration | **85.60%** | 4.9s | Optimal hyperplane separation in high-dimensional text space |

### 5. Slide / Topic 5: Live Demonstration
- Open **http://localhost:8501**
- **Step 1:** Select a sample article (e.g. *Technology* or *Science*). Click **"Classify Article"**.
- **Step 2:** Point out the **Confidence Bar Chart** showing probabilities across all 6 classes.
- **Step 3:** Switch to **"Model Dashboard"** tab:
  - Show the side-by-side **Accuracy Bar Chart**.
  - Show the **Confusion Matrix** heatmap (point out how Sports and Politics have almost zero confusion).
  - Show the **Classification Report** (Precision, Recall, F1-Score).
- **Step 4:** Switch to **"Explore Data"** tab:
  - Show the **Word Clouds** per category.
  - Show the **Top 10 Keywords** extracted directly from model coefficients.
- **Step 5 (Optional CLI Flex):** Open terminal and run:
  ```bash
  python demo_cli.py "NASA telescope spots signs of organic molecules on exoplanet"
  ```
  Show your teacher that it also works directly as a terminal tool!

---

## 💡 Top 10 Teacher Viva / Exam Questions & Answers

### Q1: Why did you choose TF-IDF over simple Bag-of-Words (CountVectorizer)?
> **Answer:** *"Bag-of-Words only counts word frequencies, which means common but uninformative words dominate the feature space. TF-IDF applies an inverse document frequency penalty to words that appear across many documents, assigning higher weights to distinctive, category-specific keywords."*

### Q2: Why did Linear SVM perform best on this text data?
> **Answer:** *"Text classification after TF-IDF vectorization results in a very high-dimensional, sparse feature space (over 15,000 features). Support Vector Machines are mathematically well-suited for high-dimensional spaces because they find the maximum-margin hyperplane separating classes, avoiding overfitting even when features outnumber samples."*

### Q3: SVMs don't natively output probabilities. How did you get confidence scores?
> **Answer:** *"We wrapped the Linear SVM inside scikit-learn's `CalibratedClassifierCV` using Platt scaling (sigmoid calibration) across 3-fold cross-validation. This maps the raw margin distance to true posterior class probabilities while preventing probability distortion."*

### Q4: How did you ensure there was no data leakage?
> **Answer:** *"We strictly wrapped both the TF-IDF vectorizer and the classifier inside an `sklearn.pipeline.Pipeline`. The vectorizer's vocabulary and IDF statistics were fitted solely on the training split (`X_train`) and only transformed on the test set (`X_test`)."*

### Q5: What is the difference between Precision and Recall in this context?
> **Answer:** 
> - *"**Precision** answers: 'Of all articles the model predicted as Sports, how many were actually Sports?'"*
> - *"**Recall** answers: 'Of all actual Sports articles in the dataset, how many did the model correctly identify?'"*
> - *"**F1-Score** is the harmonic mean of both, giving a balanced assessment."*

### Q6: Why did you use bi-grams (ngram_range=(1,2))?
> **Answer:** *"Single words often lose context. For instance, 'not good' or 'machine learning' have completely different meanings when taken together than when split into 'machine' and 'learning'. Bigrams preserve critical two-word contextual phrases."*

### Q7: Why didn't you use a heavy Deep Learning model like BERT or GPT?
> **Answer:** *"While Transformer models are state-of-the-art, for classical topic classification tasks, linear models with TF-IDF train in seconds rather than hours, require 100x fewer computational resources, have near-zero inference latency (under 10ms), and achieve comparable ~86% accuracy without needing a GPU."*

### Q8: What was the purpose of removing headers, footers, and quotes from the 20 Newsgroups dataset?
> **Answer:** *"The raw 20 Newsgroups dataset contains email headers (like `Organization:` or `From:` lines) and sign-off signatures that contain university domain names or newsgroup names. If not removed, the model would cheat by memorizing email signatures rather than learning the actual article content. Removing them forced the model to learn pure linguistic patterns."*

### Q9: What are the limitations of this project?
> **Answer:** *"First, TF-IDF does not capture semantic word order beyond adjacent bigrams (unlike sequence models like LSTMs or Transformers). Second, the 20 Newsgroups dataset has an informal conversational tone that may differ from ultra-formal modern journalism. Third, multilingual articles are not yet supported."*

### Q10: How would you improve this model in future work?
> **Answer:** *"1. Fine-tuning a lightweight pre-trained Transformer like DistilBERT for contextual embeddings. 2. Implementing active learning so the model can learn from user corrections in real time. 3. Adding web-scraping to fetch live RSS news feeds automatically."*

---

## 🚀 Presentation Checklist

- [ ] Streamlit app running in browser (`http://localhost:8501`)
- [ ] Terminal ready with `demo_cli.py` tested
- [ ] Jupyter notebook (`notebooks/exploration.ipynb`) open or available if requested
- [ ] Clean project directory structure visible in VS Code / IDE
- [ ] Confident, relaxed tone — you built a complete, working ML pipeline!
