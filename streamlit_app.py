# 결과물을 간단하게 채팅앱 형식으로 실험해볼 수 있는 스트림릿 웹사이트 코디입니다.
import streamlit as st
from streamlit_chat import message
from hyperparameters import NUM_LAYERS, D_MODEL, NUM_HEADS, DFF, DROPOUT
from utils import use_model, load_latest_checkpoint
from transformer.tensorflow import transformer
from nlp_utils import TextTokenizing


# 토크나이저를 로딩하는 함수
@st.cache(allow_output_mutation=True)
def get_tokenizer():
 
    textTokenizing = TextTokenizing()
    tokenizer = textTokenizing.load_tokenizer("super_super_small_vocab")

    VOCAB_SIZE, START_TOKEN, END_TOKEN = textTokenizing.tokens()

    return tokenizer, VOCAB_SIZE, START_TOKEN, END_TOKEN

   
# 모델을 로딩해오는 함수
@st.cache(allow_output_mutation=True)
def cached_model(VOCAB_SIZE):
    # 자신이 훈련시킨 체크포인트 디렉토리
    checkpoint_path = './training_super_small_2'
    latest_checkpoint = load_latest_checkpoint(checkpoint_path)

    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        dff=DFF,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    )

    # 웨이트를 로딩해온다.
    model.load_weights(latest_checkpoint)

    return model


tokenizer, VOCAB_SIZE, START_TOKEN, END_TOKEN = get_tokenizer()
model = cached_model(VOCAB_SIZE)

# 모델을 쉽게 사용할 수 있는 클래스를 불러오자
prediction_module = use_model(
    model=model, tokenizer=tokenizer,
    START_TOKEN=START_TOKEN, END_TOKEN=END_TOKEN, MAX_LENGTH=50
)

# HTML
# <title>Talk with A.I.</title>
st.title("Talk with A.I.")

# <header>사람같은 챗봇</header>
st.header('사람같은 챗봇')

st.markdown("[Made by Yoonhero06](https://github.com/yoonhero)")


# 세션에 대화 내용을 저장할 수 있도록 하는 코드
if 'generated' not in st.session_state:
    st.session_state["generated"] = []

if 'past' not in st.session_state:
    st.session_state["past"] = []


# 인풋 폼과 전송버튼
with st.form('form', clear_on_submit=True):
    user_input = st.text_input("You: ", "")
    submitted = st.form_submit_button("전송")


# 사용자 인풋이 전송되었을 때
if submitted and user_input:
    # 인공지능이 예측한 대답을 구한다.
    answer = prediction_module.predict(user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer)


# 화면에 로딩한다.
for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i)+'_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i],key=str(i)+"_bot")
