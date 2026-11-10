"""Lab 03 - Chaotic Text Generator: a test bench for text generators (Streamlit).
Run:  streamlit run app.py
"""
import random

import pandas as pd
import streamlit as st

from src.pipeline import (HMM, NGramGenerator, generation_metrics, load_tagged,
                          split_by_document)

st.set_page_config(page_title="Chaotic Text Generator", layout="wide")
st.title("Chaotic Text Generator - test bench")
st.warning("The corpus is synthetic and the grammar is tiny. Metrics such as valid tag pattern are "
           "defined relative to this corpus and do not measure the quality of real English.")

MODELS = {"Plain HMM": "hmm", "Role-split HMM": "hmm_split", "Trigram": "ngram"}
METRIC_LABELS = {"valid_tag_pattern": "Valid tag pattern", "plausible_subject_verb": "Plausible subject-verb",
                 "novel_sentences": "Novelty", "distinct_2": "Distinct-2"}


@st.cache_resource
def get_models():
    train, _ = split_by_document(load_tagged())
    plain = HMM(train)
    return {"hmm": plain, "hmm_split": HMM(train, split_states=True),
            "ngram": NGramGenerator([w for w, _ in train])}, train


models, train = get_models()
tagger = models["hmm"]                      # plain HMM is the tagger (Viterbi)
train_text = {" ".join(w) for w, _ in train}


def generate(key, n, temperature, seed):
    """Return a list of (words, tags, valid-state-sequence) for n sentences."""
    rng = random.Random(int(seed))
    model, out = models[key], []
    for _ in range(n):
        if key == "ngram":
            words = model.sample_sentence(rng, temperature)
            tags = tagger.viterbi(words)            # hidden tags of a trigram sentence are inferred
        else:
            words, tags = model.sample_sentence(rng, temperature)
        out.append((words, tags))
    return out


# ------------------------------------------------------------------ 1. control panel
st.header("1. Generator")
c1, c2, c3, c4 = st.columns(4)
model_name = c1.selectbox("Model", list(MODELS))
temperature = c2.slider("Temperature", 0.3, 3.0, 1.0, 0.1)
seed = c3.number_input("Random seed", 0, 10_000, 42)
n_sent = c4.slider("Number of sentences", 10, 500, 100, 10)
key = MODELS[model_name]

if st.button("Generate", type="primary"):
    st.session_state["batch"] = (model_name, temperature, seed, generate(key, n_sent, temperature, seed))

if "batch" in st.session_state:
    b_model, b_T, b_seed, batch = st.session_state["batch"]
    train_patterns = {" ".join(t) for _, t in train}
    df = pd.DataFrame({
        "sentence": [" ".join(w) for w, _ in batch],
        "tags": [" ".join(t) for _, t in batch],
    })
    df["valid pattern"] = df["tags"].isin(train_patterns).map({True: "✅", False: "❌"})
    df["copied from training"] = df["sentence"].isin(train_text).map({True: "copy", False: ""})
    st.caption(f"{len(df)} sentences - {b_model}, T={b_T}, seed={b_seed}")

    # ---- 3. metrics panel
    m = generation_metrics([(w, t) for w, t in batch], train, tagger)
    st.subheader("Metrics of this batch")
    cols = st.columns(4)
    for col, (k, label) in zip(cols, METRIC_LABELS.items()):
        col.metric(label, f"{m[k]:.3f}" if k == "distinct_2" else f"{m[k]:.1%}")
    if len(df) < 100:
        st.info("Fewer than 100 sentences: metrics are noisy. Use hundreds of sentences to judge a generator.")

    # ---- 2. table
    st.subheader("Sentences and hidden tags")
    st.dataframe(df, hide_index=True)

    # ---- 5. download
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"generated_{key}_T{b_T}_seed{b_seed}.csv", mime="text/csv")

# ------------------------------------------------------------------ optional: comparison + temperature curve
with st.expander("Optional: compare the three models at the same settings"):
    if st.button("Run comparison (500 sentences per model)"):
        rows = []
        for name, k in MODELS.items():
            res = generation_metrics(generate(k, 500, temperature, seed), train, tagger)
            rows.append({"Model": name, **{METRIC_LABELS[x]: res[x] for x in METRIC_LABELS}})
        st.dataframe(pd.DataFrame(rows).style.format({METRIC_LABELS[x]: "{:.1%}" for x in METRIC_LABELS
                                                      if x != "distinct_2"} | {"Distinct-2": "{:.3f}"}),
                     hide_index=True)

with st.expander("Optional: valid tag pattern against temperature"):
    if st.button("Plot curve (500 sentences per point)"):
        temps = [0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        curve = pd.DataFrame({name: [generation_metrics(generate(k, 500, t, seed), train, tagger)["valid_tag_pattern"]
                                     for t in temps] for name, k in MODELS.items()}, index=temps)
        curve.index.name = "temperature"
        st.line_chart(curve)

# ------------------------------------------------------------------ 4. tagger
st.header("2. Viterbi tagger")
text = st.text_input("Type an English sentence (lower-case words from the training vocabulary work best)",
                     "the cats watch the birds in the garden .")
tokens = text.lower().replace(".", " .").split()
if tokens:
    unknown = [w for w in tokens if w not in tagger.vocab]
    if unknown:
        st.warning(f"Unknown word(s), not in the training vocabulary: {', '.join(sorted(set(unknown)))}. "
                   f"Their tags are guesses and are usually unreliable.")
    pred = tagger.viterbi(tokens)
    shown = [f"{w} ⚠" if w in unknown else w for w in tokens]
    st.table(pd.DataFrame([shown, pred], index=["word", "Viterbi tag"],
                          columns=[str(i + 1) for i in range(len(tokens))]))

st.caption("A POS-HMM knows grammar patterns, not meaning: expect fluent nonsense such as "
           "'the apples sleep in the school'.")
