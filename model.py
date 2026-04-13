
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import json, math
import numpy as np
import tensorflow as tf
import keras
from keras import layers


# ── Load config + tokenizer ─────────────────────────────
def load_config():
    with open("config.json")    as f: return json.load(f)

def load_tokenizer():
    with open("tokenizer.json") as f: tok = json.load(f)
    char2idx = tok["char2idx"]
    idx2char = {int(k): v for k, v in tok["idx2char"].items()}
    return char2idx, idx2char, tok["vocab_size"]


# ── Model definition ────────────────────────────────────
def make_causal_mask(seq_len):
    mask = tf.linalg.band_part(
        tf.ones((seq_len, seq_len), dtype=tf.bool), -1, 0
    )
    return mask[tf.newaxis, tf.newaxis, :, :]


class TokenPosEmbedding(layers.Layer):
    def __init__(self, vocab_size, seq_len, embed_dim, **kw):
        super().__init__(**kw)
        self.tok_emb = layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = layers.Embedding(seq_len,   embed_dim)

    def call(self, x):
        tok = self.tok_emb(x)
        pos = self.pos_emb(tf.range(tf.shape(x)[1]))
        return tok + pos

    def compute_mask(self, inputs, mask=None):
        return None


class DecoderBlock(layers.Layer):
    def __init__(self, embed_dim, n_heads, ffn_dim, dropout=0.1, **kw):
        super().__init__(**kw)
        self.attn  = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=embed_dim // n_heads, dropout=dropout
        )
        self.ffn   = keras.Sequential([
            layers.Dense(ffn_dim,   activation="gelu"),
            layers.Dense(embed_dim, activation=None),
            layers.Dropout(dropout),
        ])
        self.ln1   = layers.LayerNormalization(epsilon=1e-6)
        self.ln2   = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, mask=None, training=False):
        n = self.ln1(x)
        x = x + self.drop1(
            self.attn(n, n, n, attention_mask=mask,
                      use_causal_mask=False, training=training),
            training=training
        )
        x = x + self.drop2(
            self.ffn(self.ln2(x), training=training),
            training=training
        )
        return x

    def compute_mask(self, inputs, mask=None):
        return None


class MiniGPT(keras.Model):
    def __init__(self, vocab_size, embed_dim, n_heads,
                 n_layers, ffn_dim, seq_len, dropout=0.1, **kw):
        super().__init__(**kw)
        self.seq_len   = seq_len
        self.embedding = TokenPosEmbedding(vocab_size, seq_len, embed_dim)
        self.emb_drop  = layers.Dropout(dropout)
        self.blocks    = [
            DecoderBlock(embed_dim, n_heads, ffn_dim, dropout, name=f"block_{i}")
            for i in range(n_layers)
        ]
        self.ln_final  = layers.LayerNormalization(epsilon=1e-6)
        self.lm_head   = layers.Dense(vocab_size, use_bias=False)

    def call(self, x, training=False):
        mask = make_causal_mask(tf.shape(x)[1])
        x    = self.emb_drop(self.embedding(x), training=training)
        for block in self.blocks:
            x = block(x, mask=mask, training=training)
        return self.lm_head(self.ln_final(x))


# ── Load model ──────────────────────────────────────────
def load_model():
    cfg                        = load_config()
    char2idx, idx2char, vsz    = load_tokenizer()

    m = MiniGPT(
        vocab_size = vsz,
        embed_dim  = cfg["EMBED_DIM"],
        n_heads    = cfg["N_HEADS"],
        n_layers   = cfg["N_LAYERS"],
        ffn_dim    = cfg["FFN_DIM"],
        seq_len    = cfg["SEQ_LEN"],
        dropout    = 0.0,
    )
    dummy = tf.zeros((1, cfg["SEQ_LEN"]), dtype=tf.int32)
    m(dummy, training=False)
    m.load_weights("minigpt_weights.weights.h5")
    m.trainable = False
    return m, char2idx, idx2char, cfg


# ── Generation ──────────────────────────────────────────
def generate(model, char2idx, idx2char, cfg,
             prompt, max_new_chars=300,
             temperature=0.8, top_k=40):

    vocab_size = cfg["VOCAB_SIZE"]
    seq_len    = cfg["SEQ_LEN"]

    ids = [char2idx[c] for c in prompt.lower() if c in char2idx]
    if not ids:
        return "No recognisable characters in prompt."

    idx = tf.constant([ids], dtype=tf.int32)

    for _ in range(max_new_chars):
        logits = model(idx[:, -seq_len:], training=False)[0, -1, :]
        logits = logits / temperature

        if top_k and top_k > 0:
            top_vals, _ = tf.math.top_k(logits, k=min(top_k, vocab_size))
            logits = tf.where(
                logits >= top_vals[-1], logits,
                tf.fill(logits.shape, float("-inf"))
            )

        probs   = tf.nn.softmax(logits).numpy()
        probs   = np.nan_to_num(probs, nan=1.0 / vocab_size)
        probs  /= probs.sum()
        next_id = np.random.choice(vocab_size, p=probs)
        idx     = tf.concat(
            [idx, tf.constant([[next_id]], dtype=tf.int32)], axis=1
        )

    prompt_len = len([c for c in prompt.lower() if c in char2idx])
    result     = "".join(idx2char[i] for i in idx[0, prompt_len:].numpy())
    return prompt.lower() + result
