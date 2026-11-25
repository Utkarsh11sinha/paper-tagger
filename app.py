"""Streamlit demo: paste a paper's title + abstract, get suggested tags from both models.

Run:  streamlit run app.py
"""
import html

import streamlit as st

from compare import comparison_markdown
from tagger import load_available

EXAMPLE_TITLE = "Vision Transformers for Dense Prediction in Medical Image Segmentation"
EXAMPLE_ABSTRACT = (
    "We study transformer-based architectures for semantic segmentation of medical images. "
    "Our model replaces the convolutional encoder of a U-Net with a vision transformer and "
    "uses self-attention to capture long-range dependencies between image patches. "
    "Experiments on MRI and CT benchmarks show improved Dice scores over convolutional "
    "baselines while requiring less labelled data through self-supervised pretraining."
)

st.set_page_config(page_title="Research Paper Tagger", page_icon="🏷️", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def get_taggers():
    return load_available()


def chips(tags):
    spans = "".join(
        f'<span style="display:inline-block;margin:4px 6px 4px 0;padding:6px 12px;'
        f'border-radius:999px;background:rgba(99,102,241,{0.15 + 0.6 * t["score"]:.2f});'
        f'border:1px solid rgba(99,102,241,.6);font-size:0.9rem">'
        f'<b>{html.escape(t["tag"])}</b> · {html.escape(t["name"])} · {t["score"]:.0%}</span>'
        for t in tags
    )
    st.markdown(spans, unsafe_allow_html=True)


def fill_example():
    st.session_state.title = EXAMPLE_TITLE
    st.session_state.abstract = EXAMPLE_ABSTRACT


st.title("🏷️ Research Paper Tagger")
st.caption("Suggests arXiv subject tags for a paper. Compares a classical ML model "
           "(TF-IDF + Logistic Regression) with a fine-tuned BERT model (MiniLM).")

taggers = get_taggers()
if not taggers:
    st.error("No trained models found in `ml_model/model/` or `dl_model/model/`. Train them with "
             "`python ml_model/train.py` and `python dl_model/train.py`.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Maximum tags", 1, 10, 5)
    use_tuned = st.checkbox("Use tuned threshold", value=True,
                            help="The threshold each model picked on the validation set.")
    manual_thr = st.slider("Threshold", 0.05, 0.95, 0.30, 0.05, disabled=use_tuned)
    st.button("Load example", on_click=fill_example)
    st.markdown(f"Models loaded: **{', '.join(t.name for t in taggers.values())}**")

title = st.text_input("Title", key="title")
abstract = st.text_area("Abstract", key="abstract", height=200)

if st.button("Suggest tags", type="primary"):
    if not (title.strip() or abstract.strip()):
        st.warning("Enter a title or an abstract first.")
    else:
        thr = None if use_tuned else manual_thr
        results = {key: tagger.suggest(title, abstract, top_k, thr) for key, tagger in taggers.items()}
        cols = st.columns(len(taggers))
        for col, (key, tags) in zip(cols, results.items()):
            with col:
                st.subheader(taggers[key].name)
                chips(tags)
                for t in tags:
                    st.progress(t["score"], text=f'{t["tag"]} — {t["score"]:.2f}')
        if len(results) == 2:
            agree = sorted({t["tag"] for t in results["ml"]} & {t["tag"] for t in results["bert"]})
            st.info("Both models agree on: " + (", ".join(agree) if agree else "no tags"))

md = comparison_markdown()
if md:
    with st.expander("Test-set metrics"):
        st.markdown(md)
