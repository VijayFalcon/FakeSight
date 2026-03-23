# FakeSight
FakeSight is an explainable fake news detector using a stacking ensemble (XGBoost + LinearSVC + LogReg) with LIME explanations and a multilingual LLM front-end. Paste any article, get a Real/Fake verdict with word-level reasoning. ⚠️ Actively developed — incremental updates incoming.

📌 Overview
FakeSight is an Explainable AI (XAI) system for fake news detection built as part of the BCSE418L Explainable Artificial Intelligence course at VIT Chennai. It goes beyond black-box classification by combining ensemble learning with interpretable explanation techniques, making every prediction transparent and human-understandable.

✨ Features

Real/Fake verdict with confidence score
LIME word-level explanation — highlights which words drove the prediction
Plain-language explanation via Groq (LLaMA 3.1) — non-technical, conversational reasoning
Multilingual support — paste articles in any language, automatic detection and translation
Gradio web interface — runs directly in Google Colab with a public shareable link


🗂️ Dataset
FakeSight is trained on a custom merged dataset combining three sources:
DatasetSamplesDescriptionWELFake72,134Benchmark English fake news datasetISOT Fake News44,898Real and fake news from Reuters and other sourcesLIAR~12,800Political statements labeled by PolitiFact
After merging, deduplication and cleaning: ~75,000 samples, ~55% real / ~45% fake.

🏗️ Architecture
Data Pipeline

Combines title + text into a single content field
TF-IDF vectorization (8,000 features)
Stratified 80/20 train/test split

Stacking Ensemble

Base Layer: XGBoost, LinearSVC, Logistic Regression trained in parallel using 5-fold out-of-fold predictions to avoid data leakage
Meta Layer: Logistic Regression that combines base model probability outputs into a final prediction
Test accuracy: ~94-95%

XAI Layer

LIME (Local Interpretable Model-agnostic Explanations) — perturbs the input and identifies the most influential words for each individual prediction
Groq LLaMA 3.1 — converts LIME output into a plain-language explanation for non-technical users

Multilingual Front-End

langdetect for fast local language detection
Groq (LLaMA 3.1) for translation of non-English articles before classification
Gradio interface for user interaction


🗃️ Repository Structure
FakeSight/
│
├── ExAIFakeNewsData.ipynb        # Data merging and preprocessing
├── ExAIFakeSightEnsemble.ipynb   # Stacking ensemble training
├── ExAIFakeSightXAI.ipynb        # Standalone SHAP + LIME analysis
├── ExAIAddLIAR.ipynb             # LIAR dataset integration
├── FakeSight.ipynb               # Main Gradio demo
└── README.md

🚀 Running the Demo

Upload all notebooks to Google Colab
Place datasets in /MyDrive/ExAI/ on Google Drive
Run ExAIFakeNewsData.ipynb to prepare the dataset
Run ExAIFakeSightEnsemble.ipynb to train and save models
Get a free Groq API key from console.groq.com
Run FakeSight.ipynb — a public Gradio link will be generated


⚠️ Known Limitations

The model was trained on datasets from 2017-2022 and exhibits temporal bias — recent breaking news with dramatic language may be misclassified as fake even when genuine. This highlights the critical importance of the XAI explanation layer, which allows users to evaluate the reasoning rather than blindly trusting the verdict.
TF-IDF has no semantic understanding — future versions will explore sentence transformers.
LIME explanations are local and stochastic — results may vary slightly between runs.


🔮 Future Work

 Retrain on recent datasets (2023-2026) to reduce temporal bias
 Replace TF-IDF with sentence transformers (e.g. all-MiniLM-L6-v2)
 Fine-tune a transformer model (RoBERTa, DistilBERT) for improved generalization
 Build a Chrome extension front-end for real-time in-browser detection
 Add user-rated explanation trust metric for human evaluation
