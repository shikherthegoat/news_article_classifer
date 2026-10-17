"""
app.py — Streamlit Web Application
====================================
Interactive dashboard for the News Article Classifier.
Run with:  streamlit run app.py
"""

import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud

from src.data_loader import load_data, CATEGORIES, CATEGORY_EMOJI, get_sample_articles
from src.model import load_all_models
from src.utils import (
    get_classification_metrics,
    get_confusion_matrix,
    get_accuracy,
    get_top_features,
)

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="News Article Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.5rem;
        background: linear-gradient(120deg, #1e3a5f, #2980b9, #1abc9c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        color: #666;
        font-size: 1.1rem;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 0.75rem;
        padding: 1.25rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached Data & Model Loading ────────────────────────────────
@st.cache_data
def cached_load_data():
    """Load and cache the dataset."""
    return load_data()


@st.cache_resource
def cached_load_models():
    """Load and cache all trained models."""
    return load_all_models()


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/newspaper.png",
        width=80,
    )
    st.title("⚙️ Settings")

    # Model selector
    model_name = st.selectbox(
        "Choose a Model",
        ["Naive Bayes", "Logistic Regression", "Linear SVM"],
        index=1,
        help="Select the ML model to use for classification.",
    )

    st.divider()

    # Info section
    st.markdown("### 📖 About")
    st.markdown(
        "This app classifies news articles into **6 categories** "
        "using NLP and Machine Learning."
    )
    st.markdown(
        "**Models:** Naive Bayes, Logistic Regression, Linear SVM"
    )
    st.markdown(
        "**Dataset:** 20 Newsgroups (scikit-learn)"
    )

    st.divider()
    st.markdown("### 🏷️ Categories")
    for cat in CATEGORIES:
        emoji = CATEGORY_EMOJI.get(cat, "📄")
        st.markdown(f"- {emoji} **{cat}**")

    st.divider()
    st.caption("Built with ❤️ using Python, scikit-learn & Streamlit")


# ── Header ─────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">'
    "<h1>📰 News Article Classifier</h1>"
    "<p>Paste a news article and let AI classify it instantly</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Load Models & Data ─────────────────────────────────────────
try:
    models = cached_load_models()
    X_train, X_test, y_train, y_test = cached_load_data()
except FileNotFoundError:
    st.error(
        "⚠️ Models not found! Please run `python train_model.py` first to train the models."
    )
    st.stop()

# ── Main Tabs ──────────────────────────────────────────────────
tab_classify, tab_dashboard, tab_explore = st.tabs([
    "🔍 Classify Article",
    "📊 Model Dashboard",
    "🔎 Explore Data",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1: CLASSIFY ARTICLE
# ═══════════════════════════════════════════════════════════════
with tab_classify:
    col_input, col_result = st.columns([3, 2], gap="large")

    with col_input:
        st.subheader("📝 Enter a News Article")

        # Sample article selector
        samples = get_sample_articles()
        sample_choice = st.selectbox(
            "Or try a sample article:",
            ["— Write your own —"] + list(samples.keys()),
        )

        if sample_choice != "— Write your own —":
            default_text = samples[sample_choice]
        else:
            default_text = ""

        article_text = st.text_area(
            "Paste your article text below:",
            value=default_text,
            height=250,
            placeholder="Enter a news article here to classify it...",
        )

        classify_btn = st.button(
            "🚀 Classify Article", type="primary", width="stretch"
        )

    with col_result:
        st.subheader("📋 Classification Result")

        if classify_btn and article_text.strip():
            model = models[model_name]

            # Predict
            prediction = model.predict([article_text])[0]
            probabilities = model.predict_proba([article_text])[0]

            # Get class order from the model
            classes = model.classes_
            prob_dict = dict(zip(classes, probabilities))

            emoji = CATEGORY_EMOJI.get(prediction, "📄")
            confidence = prob_dict[prediction] * 100

            # Result card
            st.markdown(
                f'<div class="result-card">'
                f'<h1 style="font-size:3rem; margin:0">{emoji}</h1>'
                f'<h2 style="margin:0.5rem 0">{prediction}</h2>'
                f'<p style="font-size:1.2rem; opacity:0.9">'
                f"Confidence: {confidence:.1f}%</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Confidence bar chart
            st.markdown("##### Confidence Scores by Category")
            prob_df = pd.DataFrame({
                "Category": [f"{CATEGORY_EMOJI.get(c, '')} {c}" for c in classes],
                "Confidence": [prob_dict[c] * 100 for c in classes],
            }).sort_values("Confidence", ascending=True)

            fig = px.bar(
                prob_df,
                x="Confidence",
                y="Category",
                orientation="h",
                color="Confidence",
                color_continuous_scale="Viridis",
                range_x=[0, 100],
            )
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
                coloraxis_showscale=False,
                xaxis_title="Confidence (%)",
                yaxis_title="",
            )
            fig.update_traces(
                texttemplate="%{x:.1f}%", textposition="outside"
            )
            st.plotly_chart(fig, width="stretch")

        elif classify_btn and not article_text.strip():
            st.warning("Please enter some text to classify!")
        else:
            st.info("👈 Enter an article and click **Classify** to see results.")


# ═══════════════════════════════════════════════════════════════
# TAB 2: MODEL DASHBOARD
# ═══════════════════════════════════════════════════════════════
with tab_dashboard:
    st.subheader("📊 Model Performance Dashboard")

    # ── Accuracy comparison ────────────────────────────────────
    st.markdown("#### 🏆 Model Accuracy Comparison")

    acc_data = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = get_accuracy(y_test, y_pred)
        acc_data.append({"Model": name, "Accuracy (%)": acc})

    acc_df = pd.DataFrame(acc_data)

    col_chart, col_metrics = st.columns([2, 1])

    with col_chart:
        fig = px.bar(
            acc_df,
            x="Model",
            y="Accuracy (%)",
            color="Model",
            color_discrete_sequence=["#3498db", "#2ecc71", "#e74c3c"],
            text="Accuracy (%)",
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(
            height=400,
            showlegend=False,
            yaxis_range=[0, 105],
            margin=dict(t=30),
        )
        st.plotly_chart(fig, width="stretch")

    with col_metrics:
        st.markdown("##### Quick Stats")
        for row in acc_data:
            st.metric(
                label=row["Model"],
                value=f"{row['Accuracy (%)']:.2f}%",
            )
        best = max(acc_data, key=lambda x: x["Accuracy (%)"])
        st.success(f"🥇 Best: **{best['Model']}**")

    st.divider()

    # ── Detailed metrics for selected model ────────────────────
    st.markdown(f"#### 📋 Detailed Metrics — {model_name}")

    selected_model = models[model_name]
    y_pred_selected = selected_model.predict(X_test)

    col_table, col_cm = st.columns(2)

    with col_table:
        st.markdown("##### Classification Report")
        metrics_df = get_classification_metrics(y_test, y_pred_selected, CATEGORIES)
        st.dataframe(
            metrics_df.style.format({
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1-Score": "{:.3f}",
                "Support": "{:.0f}",
            }).background_gradient(
                cmap="YlGn", subset=["Precision", "Recall", "F1-Score"]
            ),
            width="stretch",
            height=380,
        )

    with col_cm:
        st.markdown("##### Confusion Matrix")
        cm_df = get_confusion_matrix(y_test, y_pred_selected, CATEGORIES)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm_df,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CATEGORIES,
            yticklabels=CATEGORIES,
            ax=ax,
            linewidths=0.5,
            square=True,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════════════════
# TAB 3: EXPLORE DATA
# ═══════════════════════════════════════════════════════════════
with tab_explore:
    st.subheader("🔎 Data Exploration")

    # ── Dataset distribution ───────────────────────────────────
    st.markdown("#### 📈 Category Distribution")

    col_train, col_test = st.columns(2)

    with col_train:
        st.markdown("##### Training Set")
        train_counts = pd.Series(y_train).value_counts().sort_index()
        fig = px.pie(
            values=train_counts.values,
            names=[f"{CATEGORY_EMOJI.get(c, '')} {c}" for c in train_counts.index],
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        fig.update_layout(height=350, margin=dict(t=20, b=20))
        fig.update_traces(textinfo="percent+value")
        st.plotly_chart(fig, width="stretch")

    with col_test:
        st.markdown("##### Test Set")
        test_counts = pd.Series(y_test).value_counts().sort_index()
        fig = px.pie(
            values=test_counts.values,
            names=[f"{CATEGORY_EMOJI.get(c, '')} {c}" for c in test_counts.index],
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        fig.update_layout(height=350, margin=dict(t=20, b=20))
        fig.update_traces(textinfo="percent+value")
        st.plotly_chart(fig, width="stretch")

    st.divider()

    # ── Word Clouds ────────────────────────────────────────────
    st.markdown("#### ☁️ Word Clouds by Category")
    st.caption("Most frequently occurring words in each news category")

    # Combine all training text by category
    category_texts = {}
    for text, label in zip(X_train, y_train):
        category_texts.setdefault(label, []).append(text)

    # Display word clouds in a 3×2 grid
    wc_cols = st.columns(3)
    colors = ["Blues", "Greens", "Oranges", "Purples", "Reds", "YlGn"]

    for i, category in enumerate(CATEGORIES):
        with wc_cols[i % 3]:
            emoji = CATEGORY_EMOJI.get(category, "📄")
            st.markdown(f"##### {emoji} {category}")

            combined = " ".join(category_texts.get(category, [""]))
            if combined.strip():
                wc = WordCloud(
                    width=400,
                    height=250,
                    background_color="white",
                    colormap=colors[i % len(colors)],
                    max_words=80,
                    contour_width=1,
                    contour_color="steelblue",
                    collocations=False,
                ).generate(combined)

                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                plt.tight_layout(pad=0)
                st.pyplot(fig)
                plt.close()

    st.divider()

    # ── Top features ───────────────────────────────────────────
    st.markdown("#### 🔑 Top Keywords per Category")
    st.caption(f"Most important words for **{model_name}**")

    selected_model = models[model_name]
    vectorizer = selected_model.named_steps["tfidf"]
    clf = selected_model.named_steps["clf"]
    top_feats = get_top_features(vectorizer, clf, CATEGORIES, n=10)

    if top_feats:
        feat_cols = st.columns(3)
        for i, (cat, words) in enumerate(top_feats.items()):
            with feat_cols[i % 3]:
                emoji = CATEGORY_EMOJI.get(cat, "📄")
                st.markdown(f"**{emoji} {cat}**")
                for j, word in enumerate(words, 1):
                    st.markdown(f"`{j}.` {word}")
    else:
        st.info("Feature extraction not available for this model type.")
