# Customer Support Utterance Clustering & Embedding Fine-Tuning

Unsupervised clustering of 26,872 customer support utterances using sentence embeddings, UMAP dimensionality reduction, and HDBSCAN clustering. Includes fine-tuning the embedding model with intent supervision to improve cluster quality.

**Dataset:** [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (27 intents, 11 categories)

## Repository Structure

```text
Contact_Center/
├── eda_customer_support.ipynb      # Exploratory data analysis
├── clustering_analysis.ipynb       # Clustering pipeline & evaluation
├── finetune_embeddings.ipynb       # Embedding model fine-tuning
├── finetune_methodology.ipynb      # Fine-tuning methodology documentation
├── main.py                         # Minimal entry point
├── pyproject.toml                  # Project dependencies
├── uv.lock                         # Locked dependency versions
├── .python-version                 # Python 3.13
├── .gitignore                      # Excludes .env, checkpoints, model artifacts
└── .env                            # API keys (not committed)
```

## Environment Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and Python 3.13.

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or via Homebrew
brew install uv
```

### 2. Create the virtual environment and install dependencies

```bash
cd Contact_Center
uv sync
```

This reads `pyproject.toml` and `uv.lock`, creates a `.venv` directory, and installs all pinned dependencies.

### 3. Register the Jupyter kernel

```bash
uv run python -m ipykernel install --user --name contact-center --display-name "Contact Center (Python 3.13)"
```

### 4. Set up API keys

Create a `.env` file in the project root with your Anthropic API key (required only for LLM-based cluster labeling in `clustering_analysis.ipynb`):

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Launch Jupyter

Open any notebook in VS Code (select the **Contact Center (Python 3.13)** kernel) or launch Jupyter directly:

```bash
uv run jupyter lab
```

## Notebooks

### 1. `eda_customer_support.ipynb` — Exploratory Data Analysis

**Run this first** to understand the dataset before clustering.

- Loads the Bitext dataset from HuggingFace
- Dataset shape, dtypes, memory usage, missing values, duplicates
- Category distribution (11 classes) and intent distribution (27 classes)
- Per-category intent breakdown with heatmaps
- Text length analysis (character and word count distributions)
- Word frequency analysis and word clouds
- Response diversity metrics per intent

### 2. `clustering_analysis.ipynb` — Clustering Pipeline & Evaluation

**Core analysis notebook.** Embeds utterances and discovers clusters without using ground-truth labels, then evaluates against them.

| Step | What | Tool |
| ---- | ---- | ---- |
| 1 | Embed utterances | `all-MiniLM-L6-v2` (384-dim sentence transformer) |
| 2 | Dimensionality reduction | UMAP (15d for clustering, 2d for visualization) |
| 3 | Cluster | HDBSCAN with parameter sweep over `min_cluster_size` |
| 4 | Evaluate | ARI & NMI vs ground-truth intent (27) and category (11) |
| 5 | Label clusters | TF-IDF top terms + LLM labeling via Claude API |

**Key outputs:**

- UMAP scatter plots colored by ground-truth labels and discovered clusters
- Contingency matrices (intent vs cluster, category vs cluster)
- Per-intent cluster purity analysis
- TF-IDF and LLM-generated cluster labels with summary table

**Note:** The LLM labeling section requires a valid `ANTHROPIC_API_KEY` in `.env`.

### 3. `finetune_embeddings.ipynb` — Embedding Fine-Tuning

**Fine-tunes the embedding model** to improve clustering quality by pulling same-intent utterances closer together in the embedding space.

| Step | What |
| ---- | ---- |
| 1 | Stratified 80/20 train/val split by intent |
| 2 | Generate ~27K (anchor, positive) same-intent pairs |
| 3 | Fine-tune with `MultipleNegativesRankingLoss` (3 epochs, batch 64) |
| 4 | Re-run identical UMAP + HDBSCAN pipeline on both base and fine-tuned embeddings |
| 5 | Compare ARI, NMI, per-intent purity, and embedding separation |

**Key outputs:**

- Side-by-side UMAP plots (base vs fine-tuned)
- ARI/NMI comparison bar charts
- Per-intent purity comparison
- Intra-class vs inter-class cosine similarity histograms
- Fine-tuned model saved to `./finetuned-MiniLM-L6-v2-customer-support/`

### 4. `finetune_methodology.ipynb` — Methodology Documentation

**Reference document** (all Markdown, no code). Explains the fine-tuning approach in detail:

- Problem statement and motivation
- Dataset and base model properties
- Train/val split strategy and data leakage prevention
- Pair and triplet generation methodology
- MultipleNegativesRankingLoss (InfoNCE) explanation with illustrations
- Hyperparameter choices and rationale
- Evaluation methodology and metric definitions
- Design decisions and trade-offs
- Reproducibility instructions
- Potential improvements

## Recommended Execution Order

```text
1. eda_customer_support.ipynb        → understand the data
2. clustering_analysis.ipynb         → baseline clustering results
3. finetune_embeddings.ipynb         → fine-tune and compare
4. finetune_methodology.ipynb        → reference documentation (read-only)
```
