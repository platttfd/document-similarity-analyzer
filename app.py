import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="DocStruct | Linear Algebra Document Analyzer",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner UI cards and spacing
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    input_mode = st.radio(
        "Data Source",
        ["Built-in Demo Corpus (1-Click Viva Demo)", "Upload .txt Files"],
        index=0
    )
    
    st.divider()
    st.subheader("Preprocessing & Math")
    min_word_len = st.slider("Minimum Word Length", min_value=1, max_value=6, value=3)
    use_binary = st.checkbox("Use Binary Incidence Matrix (0/1)", value=False, help="Check to count presence (1) vs raw term frequency.")
    custom_stopwords = st.text_input(
        "Extra Stopwords (comma-separated)", 
        value="and, are, for, in, of, the, is, to, with, on, at, by, an, be"
    )
    
    st.divider()
    st.caption("Topic 41 • Linear Algebra Vector Space & PCA Structural Analyzer")

# --- MAIN HEADER ---
st.title("📐 Document Similarity & Structural Analyzer")
st.markdown("Explore **Vector Space Models**, **Matrix Rank**, **Cosine Similarity**, and **2D Principal Component Analysis (PCA)**.")

# --- DATA LOADING ---
documents = []
doc_labels = []

if input_mode == "Built-in Demo Corpus (1-Click Viva Demo)":
    sample_docs = {
        "AI_Algorithms.txt": "Artificial intelligence and machine learning algorithms transform modern data analysis and predictive modeling.",
        "Neural_Networks.txt": "Deep learning neural networks power computer vision, natural language processing, and speech recognition systems.",
        "ML_Foundations.txt": "Machine learning models require clean data, feature engineering, and strong mathematical foundations.",
        "Linear_Algebra.txt": "Linear algebra, matrices, eigenvalues, and eigenvectors are essential mathematical tools for data science.",
        "Vector_Spaces.txt": "Vector spaces, norms, orthogonality, and inner products define geometric relationships in mathematics.",
        "Matrix_Rank.txt": "Matrix rank and linear independence determine the true dimensionality of subspaces and transformations.",
        "Cyber_Security.txt": "Computer networks, cybersecurity protocols, encryption, and firewalls protect digital infrastructure.",
        "Wellness_Health.txt": "Daily physical exercise, balanced nutrition, morning routines, and proper hydration improve health and focus."
    }
    doc_labels = list(sample_docs.keys())
    documents = list(sample_docs.values())
    st.info("💡 **Demo Mode Active:** Analyzing 8 pre-loaded topic documents across AI, Linear Algebra, Cybersecurity, and Wellness. Switch to **Upload .txt Files** in the sidebar to test your own files.")
else:
    uploaded_files = st.file_uploader(
        "Drag and drop at least 3 text (.txt) files", 
        type=["txt"], 
        accept_multiple_files=True
    )
    if uploaded_files:
        for file in uploaded_files:
            content = file.read().decode("utf-8", errors="ignore").strip()
            if content:
                documents.append(content)
                doc_labels.append(file.name)

# --- VALIDATION & COMPUTATION ---
if len(documents) < 3:
    st.warning("⚠️ Please provide at least **3 non-empty documents** to construct the vector space and compute 2D PCA projections.")
    st.stop()

# Tokenization & Vocabulary
stop_words = {w.strip().lower() for w in custom_stopwords.split(",") if w.strip()}
tokenized_docs = []
vocab_set = set()

for doc in documents:
    clean_doc = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in doc)
    words = [w for w in clean_doc.split() if w not in stop_words and len(w) >= min_word_len]
    tokenized_docs.append(words)
    vocab_set.update(words)

vocabulary = sorted(list(vocab_set))

if len(vocabulary) < 2:
    st.error("Vocabulary is too small after filtering. Lower the minimum word length or reduce stopwords.")
    st.stop()

# Term-Document Matrix A (Terms x Documents)
A = np.zeros((len(vocabulary), len(documents)), dtype=float)
for col_idx, words in enumerate(tokenized_docs):
    for word in words:
        row_idx = vocabulary.index(word)
        if use_binary:
            A[row_idx, col_idx] = 1.0
        else:
            A[row_idx, col_idx] += 1.0

# Linear Algebra Metrics
rank_A = int(np.linalg.matrix_rank(A))
num_docs = A.shape[1]
norms = np.linalg.norm(A, axis=0)

# Cosine Similarity Matrix
sim_matrix = np.zeros((num_docs, num_docs))
for i in range(num_docs):
    for j in range(num_docs):
        if norms[i] > 0 and norms[j] > 0:
            sim_matrix[i, j] = np.dot(A[:, i], A[:, j]) / (norms[i] * norms[j])

# Find Top Matching Pair (excluding self-similarity on diagonal)
sim_no_diag = sim_matrix.copy()
np.fill_diagonal(sim_no_diag, -1.0)
max_idx = np.unravel_index(np.argmax(sim_no_diag), sim_no_diag.shape)
top_score = sim_no_diag[max_idx]
top_pair = f"{doc_labels[max_idx[0]]} ↔ {doc_labels[max_idx[1]]}"

# PCA via Covariance Eigen-decomposition
X = A.T  # Documents as rows, Terms as features
X_centered = X - np.mean(X, axis=0)
cov_matrix = np.cov(X_centered, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

pcs = eigenvectors[:, :2]
total_var = np.sum(eigenvalues)
exp_var = (eigenvalues[:2] / total_var) if total_var > 0 else np.array([0.0, 0.0])
Z_proj = np.dot(X_centered, pcs)

# --- KPI METRICS ROW ---
st.subheader("📊 Structural Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Documents (n)", num_docs)
c2.metric("Vocabulary Terms (m)", len(vocabulary))
c3.metric("Matrix Rank(A)", f"{rank_A} / {min(len(vocabulary), num_docs)}")
c4.metric("2D PCA Variance", f"{np.sum(exp_var)*100:.1f}%")

if top_score > 0:
    st.success(f"🔗 **Closest Document Pair:** `{top_pair}` with a Cosine Similarity of **{top_score:.3f}**")
else:
    st.info("ℹ️ All documents are completely orthogonal (Cosine Similarity = 0.000 across all distinct pairs).")

# --- INTERACTIVE TABS ---
tab_viz, tab_tdm, tab_sim, tab_pca, tab_inspector = st.tabs([
    "📈 Visual Plots", 
    "🔢 Term-Document Matrix", 
    "🤝 Cosine Similarity", 
    "🧭 PCA Coordinates",
    "📄 Document Inspector"
])

with tab_viz:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # Heatmap
    im = axes[0].imshow(sim_matrix, cmap="YlGnBu", vmin=0, vmax=1)
    axes[0].set_xticks(range(num_docs))
    axes[0].set_yticks(range(num_docs))
    axes[0].set_xticklabels(doc_labels, rotation=40, ha="right", fontsize=9)
    axes[0].set_yticklabels(doc_labels, fontsize=9)
    axes[0].set_title("Pairwise Cosine Similarity Heatmap", fontweight="bold", pad=12)
    
    # Annotate values inside heatmap cells
    for i in range(num_docs):
        for j in range(num_docs):
            val = sim_matrix[i, j]
            text_color = "white" if val > 0.55 else "black"
            axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)
            
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # PCA Scatter
    axes[1].scatter(Z_proj[:, 0], Z_proj[:, 1], color="#1f77b4", s=110, edgecolors="black", zorder=3)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.9)
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=0.9)
    for i, label in enumerate(doc_labels):
        axes[1].annotate(
            label, 
            (Z_proj[i, 0], Z_proj[i, 1]), 
            textcoords="offset points", 
            xytext=(0, 9), 
            ha="center", 
            fontsize=8.5, 
            fontweight="bold"
        )
    axes[1].set_title("2D PCA Subspace Projection", fontweight="bold", pad=12)
    axes[1].set_xlabel(f"PC1 ({exp_var[0]*100:.1f}% variance)")
    axes[1].set_ylabel(f"PC2 ({exp_var[1]*100:.1f}% variance)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    st.pyplot(fig)

with tab_tdm:
    st.markdown(f"**Matrix Dimensions:** `{len(vocabulary)} terms × {num_docs} documents`")
    tdm_df = pd.DataFrame(A.astype(int), index=vocabulary, columns=doc_labels)
    st.dataframe(tdm_df, use_container_width=True)
    st.download_button(
        "⬇️ Download Term-Document Matrix (CSV)",
        data=tdm_df.to_csv().encode("utf-8"),
        file_name="term_document_matrix.csv",
        mime="text/csv"
    )

with tab_sim:
    st.markdown("**Pairwise Cosine Similarity:** $\\cos(\\theta) = \\frac{d_i \\cdot d_j}{\\|d_i\\| \\|d_j\\|}$")
    sim_df = pd.DataFrame(np.round(sim_matrix, 4), index=doc_labels, columns=doc_labels)
    st.dataframe(sim_df.style.background_gradient(cmap="YlGnBu", vmin=0, vmax=1), use_container_width=True)
    st.download_button(
        "⬇️ Download Similarity Matrix (CSV)",
        data=sim_df.to_csv().encode("utf-8"),
        file_name="cosine_similarity.csv",
        mime="text/csv"
    )

with tab_pca:
    st.markdown("**Principal Component Coordinates & Eigenvalue Spectrum**")
    pca_df = pd.DataFrame({
        "PC1": np.round(Z_proj[:, 0], 4),
        "PC2": np.round(Z_proj[:, 1], 4),
        "Vector L2 Norm": np.round(norms, 4)
    }, index=doc_labels)
    st.dataframe(pca_df, use_container_width=True)

with tab_inspector:
    selected_doc = st.selectbox("Select a document to inspect its extracted tokens and vector:", doc_labels)
    doc_idx = doc_labels.index(selected_doc)
    
    st.markdown("**Original Text:**")
    st.info(documents[doc_idx])
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Extracted Terms ({len(tokenized_docs[doc_idx])} tokens):**")
        st.write(", ".join(tokenized_docs[doc_idx]))
    with col_b:
        st.markdown("**Non-Zero Vector Weights:**")
        non_zero = {vocabulary[r]: int(A[r, doc_idx]) for r in range(len(vocabulary)) if A[r, doc_idx] > 0}
        st.json(non_zero)
