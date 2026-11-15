import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Document Similarity & Structural Analyzer", layout="wide")

st.title("Topic 41: Document Similarity and Structural Analyzer")
st.markdown("Analyze vector spaces, term-document matrices, cosine similarity, and 2D PCA projections.")

# Default text corpus
default_input = """Artificial intelligence and machine learning algorithms transform data analysis.
Deep learning neural networks power modern computer vision and speech systems.
Machine learning models require clean data and strong mathematical foundations.
Linear algebra, matrices, and eigenvalues are essential for data science.
Vector spaces, norms, and inner products define geometric relationships in math.
Matrix rank and linear independence determine the dimensionality of vector spaces.
Python programming is widely used for software engineering and web development.
Software developers write clean code and test algorithms for scalable systems.
Computer networks, cybersecurity, and protocols protect digital infrastructure.
Daily exercise, balanced nutrition, and proper hydration improve physical health.
Healthy morning routines and wellness habits boost daily energy and focus.
Football and basketball require teamwork, endurance, and strategic gameplay."""

raw_text = st.text_area("Input Documents (One document per line)", value=default_input, height=200)

documents = [line.strip() for line in raw_text.split("\n") if line.strip()]

if len(documents) < 3:
    st.warning("Please enter at least 3 documents to perform matrix and PCA analysis.")
else:
    doc_labels = [f"Doc {i+1}" for i in range(len(documents))]
    
    # Feature Extraction (Term-Document Matrix)
    stop_words = {"and", "are", "for", "in", "of", "the", "is", "to"}
    tokenized_docs = []
    vocab_set = set()
    for doc in documents:
        clean_doc = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in doc)
        words = [w for w in clean_doc.split() if w not in stop_words and len(w) > 2]
        tokenized_docs.append(words)
        vocab_set.update(words)
        
    vocabulary = sorted(list(vocab_set))
    A = np.zeros((len(vocabulary), len(documents)), dtype=float)
    for col_idx, words in enumerate(tokenized_docs):
        for word in words:
            A[vocabulary.index(word), col_idx] += 1.0
            
    # Linear Algebra Computations
    rank_A = np.linalg.matrix_rank(A)
    num_docs = A.shape[1]
    norms = np.array([np.sqrt(np.dot(A[:, j], A[:, j])) for j in range(num_docs)])
    sim_matrix = np.zeros((num_docs, num_docs))
    for i in range(num_docs):
        for j in range(num_docs):
            if norms[i] > 0 and norms[j] > 0:
                sim_matrix[i, j] = np.dot(A[:, i], A[:, j]) / (norms[i] * norms[j])
                
    # PCA Projection
    X = A.T
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
    pcs = eigenvectors[:, :2]
    exp_var = eigenvalues[:2] / np.sum(eigenvalues)
    Z_proj = np.dot(X_centered, pcs)

    # Metrics Display
    st.markdown("### Structural Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Documents", num_docs)
    col2.metric("Vocabulary Size", len(vocabulary))
    col3.metric("Matrix Rank", rank_A)
    col4.metric("PCA Variance Retained", f"{np.sum(exp_var)*100:.1f}%")

    # Visualizations
    st.markdown("### Visualizations")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(sim_matrix, cmap="YlGnBu", vmin=0, vmax=1)
    axes[0].set_xticks(range(num_docs))
    axes[0].set_yticks(range(num_docs))
    axes[0].set_xticklabels(doc_labels, rotation=45)
    axes[0].set_yticklabels(doc_labels)
    axes[0].set_title("Cosine Similarity Matrix")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    
    axes[1].scatter(Z_proj[:, 0], Z_proj[:, 1], color="#2b5c8f", s=90, edgecolors="black", zorder=3)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    for i, label in enumerate(doc_labels):
        axes[1].annotate(label, (Z_proj[i, 0], Z_proj[i, 1]), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold")
    axes[1].set_title(f"2D PCA Projection")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig)

    # DataFrames
    tab1, tab2 = st.tabs(["Similarity Matrix", "PCA Coordinates"])
    with tab1:
        st.dataframe(pd.DataFrame(np.round(sim_matrix, 3), index=doc_labels, columns=doc_labels))
    with tab2:
        st.dataframe(pd.DataFrame(np.round(Z_proj, 4), index=doc_labels, columns=["PC1", "PC2"]))
