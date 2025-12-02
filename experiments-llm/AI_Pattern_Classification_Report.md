
# AI Design Pattern Classification Report

## 1. Introduction
This report summarizes the classification of AI Design Patterns using various embedding techniques and machine learning models.

## 2. Dataset
- **Source**: /root/AI-Pattern-Mining-Project/outputs/prompt & rag/20251028_085918 - Run/generated_code-v2
- **Samples**: 1430
- **Classes**: 18 (Advanced LLM Prompting, Cross-lingual LLM Prompting, Enhanced User Intent Comprehension with LLMs, Explainable AI (XAI) Techniques, Integrating External Knlowladge with LLM, LLM Agent Training & Alignment, LLM Code Execution for Precision, LLM Context Management, LLM KV Cache Optimization, LLM Results Evaluation, LLM based Multimodal Generative Prompting, LLM based Planning, Iterative Optimizations and ReAct, or Reasoning, Think Step by step, XoT, LLMs for Recommender Systems, Modular LLM Agent Architectures, Reliable, Transparent, & Augmented LLMs, Retrieval Augmented Generation(RAG) Optimization for LLMs, Structured Output & Formatting for LLMs, Tool Use for LLMs)

## 3. Embeddings Evaluated
- Traditional: TF-IDF, Word2Vec, GloVe
- Modern: Sentence-BERT, CodeBERT, RoBERTa, Jina-V2 (where available)

## 4. Models Evaluated
- **ML**: Logistic Regression, Naive Bayes, SVM, Random Forest, Gradient Boosting, XGBoost, KNN
- **DL**: Simple NN, CNN, LSTM, BiLSTM

## 5. Results Summary
Top 5 Performing Models:
| Model               | Embedding   |   Accuracy |   F1-score |
|:--------------------|:------------|-----------:|-----------:|
| SVM                 | TF-IDF      |   0.723776 |   0.716944 |
| Logistic Regression | TF-IDF      |   0.72028  |   0.712002 |
| Logistic Regression | Nomic       |   0.688811 |   0.695305 |
| XGBoost             | TF-IDF      |   0.699301 |   0.694366 |
| KNN                 | TF-IDF      |   0.692308 |   0.682468 |

## 6. Explainability
LIME analysis was performed to identify key tokens contributing to classification.

## 7. Conclusion
The best performing model was **SVM** with **TF-IDF** embedding, achieving an F1-score of **0.7169**.
