import streamlit as st
from streamlit_chat import message
import pandas as pd
import numpy as np
from hyperparameters import NUM_LAYERS, D_MODEL, NUM_HEADS, DFF, DROPOUT
from utils import use_model, load_latest_checkpoint
from transformer import transformer
from nlp_utils import TextTokenizing


@st.cache(allow_output_mutation=True)
def get_tokenizer():
 
    textTokenizing = TextTokenizing()
    tokenizer = textTokenizing.load_tokenizer("super_super_small_vocab")

    VOCAB_SIZE, START_TOKEN, END_TOKEN = textTokenizing.tokens()

    return tokenizer, VOCAB_SIZE, START_TOKEN, END_TOKEN

   

@st.cache(allow_output_mutation=True)
def cached_model(VOCAB_SIZE):
    checkpoint_path = './training_super_small'
    latest_checkpoint = load_latest_checkpoint(checkpoint_path)

    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        dff=DFF,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    )

    model.load_weights(latest_checkpoint)

    return model


tokenizer, VOCAB_SIZE, START_TOKEN, END_TOKEN = get_tokenizer()
model = cached_model(VOCAB_SIZE)
prediction_module = use_model(
    model=model, tokenizer=tokenizer,
    START_TOKEN=START_TOKEN, END_TOKEN=END_TOKEN, MAX_LENGTH=50
)

st.title("Talk with A.I.")

st.header('사람같은 챗봇')
st.markdown("[Made by Yoonhero06](https://github.com/yoonhero)")

if 'generated' not in st.session_state:
    st.session_state["generated"] = []

if 'past' not in st.session_state:
    st.session_state["past"] = []


with st.form('form', clear_on_submit=True):
    user_input = st.text_input("You: ", "")
    submitted = st.form_submit_button("전송")


if submitted and user_input:
    answer = prediction_module.predict(user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer)


for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i)+'_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i],key=str(i)+"_bot")
