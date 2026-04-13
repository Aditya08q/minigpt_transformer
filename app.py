
import streamlit as st
from model import load_model, generate

st.set_page_config(
    page_title = "Mini GPT",
    page_icon  = "",
    layout     = "wide"
)

#  Load once 
@st.cache_resource
def get_model():
    return load_model()

model, char2idx, idx2char, cfg = get_model()

#  Sidebar 
with st.sidebar:
    st.title("Settings")
    temperature = st.slider("Temperature", 0.3, 1.5, 0.8, 0.05)
    max_chars   = st.slider("Max characters", 50, 500, 250, 10)
    top_k       = st.slider("Top-K", 0, cfg["VOCAB_SIZE"], 40, 1)

    st.divider()
    st.subheader("Model info")
    st.metric("Parameters", "4.8M")
    st.metric("Train loss",  f"{cfg['final_train_loss']:.4f}")
    st.metric("Val loss",    f"{cfg['final_val_loss']:.4f}")
    st.metric("Vocab size",  cfg["VOCAB_SIZE"])
    st.metric("Layers",      cfg["N_LAYERS"])

    st.divider()
    st.markdown("""
| Temp | Style |
|------|-------|
| 0.5–0.7 | Safe |
| 0.8–0.9 | Best  |
| 1.0–1.2 | Creative |
| >1.2 | Chaotic |
    """)

# ── Main ────────────────────────────────────────────────
st.title(" Mini GPT — Story Generator")
st.caption("4.8M param transformer · TensorFlow + Keras 3 · trained from scratch")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Prompt")

    # Quick prompts
    quick_prompts = [
        "once upon a time there was a little girl named lily",
        "tom was sad because he lost his favorite toy",
        "the dog and the cat became best friends",
        "one day a kind old man found a magic stone",
    ]
    for q in quick_prompts:
        if st.button(q[:48] + "...", use_container_width=True):
            st.session_state["prompt"] = q

    prompt = st.text_area(
        "Or type your own prompt:",
        value  = st.session_state.get("prompt",
                 "once upon a time there was a little girl named lily"),
        height = 120,
    )
    go = st.button("Generate Story", type="primary",
                   use_container_width=True)

with col2:
    st.subheader("Generated story")
    if go:
        if not prompt.strip():
            st.warning("Enter a prompt first.")
        else:
            with st.spinner("Generating..."):
                result = generate(
                    model, char2idx, idx2char, cfg,
                    prompt      = prompt,
                    max_new_chars = max_chars,
                    temperature   = temperature,
                    top_k         = top_k if top_k > 0 else None,
                )
            st.text_area("", value=result, height=320,
                         label_visibility="collapsed")
            c1, c2, c3 = st.columns(3)
            c1.metric("Generated", f"{len(result)-len(prompt)} chars")
            c2.metric("Temp", temperature)
            c3.metric("Top-K", top_k if top_k > 0 else "off")
    else:
        st.info("Pick a prompt and click Generate Story.")
