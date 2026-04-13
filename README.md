#  Mini GPT Transformer (From Scratch)

##  Overview

This project implements a **Mini GPT (decoder-only Transformer)** model from scratch for text generation.
The model is trained on the TinyStories dataset and can generate short, coherent story-like text.

The goal of this project is to understand how transformer-based language models work and build a complete pipeline from data processing to deployment.

---

## Features

* Built a **GPT-style Transformer architecture from scratch**
* Implemented **self-attention mechanism** and **causal masking**
* Custom **character-level tokenizer**
* Autoregressive **text generation (next token prediction)**
* Supports **temperature and top-k sampling**
* Simple **UI for text generation** (Streamlit / Gradio)

---

## Architecture

Input Text
→ Tokenization (character-level)
→ Embedding + Positional Encoding
→ Transformer Decoder Blocks
→ Linear Layer + Softmax
→ Next Token Prediction
→ Generated Text

---

## Model Details

* Architecture: Decoder-only Transformer
* Layers: 6
* Parameters: ~4.8M
* Vocabulary Size: 54
* Training Dataset: TinyStories
* Framework: TensorFlow / Keras (or PyTorch — update based on your version)

---

## Data Processing

* Cleaned raw text (lowercasing, whitespace normalization)
* Removed noisy/short samples
* Built custom vocabulary
* Converted text into integer token sequences

---

##  Training

* Loss Function: Cross-Entropy
* Optimizer: Adam / AdamW
* Sequence Length: 128
* Training on Google Colab (Free GPU)

---

## Text Generation

The model generates text using:

* Autoregressive decoding
* Temperature sampling
* Top-K sampling

Example:
Input: once upon a time
Output: once upon a time there was a little girl...

---

##  Run the Project

###  Install dependencies
pip install tensorflow streamlit numpy

### 2️ Run UI
streamlit run app.py


---

## Model Weights

Due to size constraints, model weights are hosted externally:
 [Download Weights](YOUR_GOOGLE_DRIVE_LINK)

---

## Future Improvements

* Switch to word/subword tokenization
* Improve long-range coherence
* Add top-p (nucleus) sampling
* Scale model size

---

## Acknowledgment
This project is built for learning and understanding decoder_based transformer architectures.

