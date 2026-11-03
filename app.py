# Copyright (c) 2026 Shubhan Punde · https://github.com/Drakeion2532/cyberbullying-detection · MIT licence · sp2532-f30cd46f33af
"""Streamlit app: paste a message, see the category, confidence and each model's vote.

    streamlit run app.py
"""
import pandas as pd
import streamlit as st

from cbd import MODELS
from cbd.predict import LABELS, classify, load

st.set_page_config(page_title="Cyberbullying detector", layout="centered")
st.title("Cyberbullying detector")
st.caption("Five models (LightGBM, XGBoost, SVM, Random Forest, Logistic Regression) vote on each message.")

task = st.radio("Categories", ["five_class", "six_class"], horizontal=True,
                format_func=lambda t: "Targeted (5)" if t == "five_class" else "Including 'other' (6)")
if not (MODELS / f"ensemble_{task}.joblib").exists():
    st.warning(f"No trained model yet. Run `python -m cbd.train --task {task}` first.")
    st.stop()


@st.cache_resource
def get(task):
    return load(task)


text = st.text_area("Message", height=120, placeholder="Paste a tweet, comment or chat message")
if st.button("Check", type="primary") and text.strip():
    r = classify(get(task), text)
    bullying = r["label"] != "not_cyberbullying"
    (st.error if bullying else st.success)(f"**{LABELS[r['label']]}**: {r['confidence']:.0%} confidence")
    st.bar_chart(pd.Series(r["probabilities"], name="probability"))
    st.write("Each model's vote:", r["votes"])
    st.caption("A screening aid, not a verdict: sarcasm, quotes and reclaimed words can fool it. "
               "A person should review anything that leads to action.")
