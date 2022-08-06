import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from nlp_utils import preprocess_sentence, TextTokenizing
from transformer import transformer, CustomSchedule, loss_function
from utils import make_checkpoint, accuracy, load_csv_and_processing
from hyperparameters import NUM_LAYERS, D_MODEL, NUM_HEADS, DFF, DROPOUT


# Loading DataSet
# CSV Structure Example
# Q                |                 A 
# 뭐해?            |               코딩!       
questions, answers = load_csv_and_processing("./small_dataset.csv")


questions = questions[20000:-5000]
answers = answers[20000:-5000]


# 토크나이저 모듈 
textTokenizing = TextTokenizing()

# 토크나이저를 생성한다.
# tokenizer = textTokenizing.create_tokenizer(questions, answers, target_vocab_size=2**15)

# 토크나이저를 저장해서 다음 실행 때 시간을 단축하고 싶다면 사용
# textTokenizing.save_tokenizer("super_super_3_small_vocab")

# 토크나이저를 로딩
textTokenizing.load_tokenizer("super_super_3_small_vocab")


# 보캡 사이즈, 스타트 토큰, 엔드 토큰을 불러온다.
VOCAB_SIZE, START_TOKEN, END_TOKEN = textTokenizing.tokens()

print(f"보캡 사이즈: {VOCAB_SIZE}, 스타트 토큰: {START_TOKEN}, 엔드 토큰: {END_TOKEN}")


# 입력된 문장을 토크나이징한다.
questions, answers = textTokenizing.tokenize_and_filter(questions, answers)

print(f'질문 데이터의 크기:{questions.shape}')
print(f'답변 데이터의 크기:{answers.shape}')


# 배치 크기와 버퍼 사이즈를 선언
BATCH_SIZE = 64
BUFFER_SIZE = 20000

# 훈련을 위한 데이터셋을 생성한다.
dataset = textTokenizing.make_dataset(BATCH_SIZE, BUFFER_SIZE)


# 트랜스포머 모델을 선언한다.
model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)


# 모델의 전체적인 구조를 보여준다.
model.summary()


# 콜백 함수를 만들어서 웨이트 값이 훈련하면서 저장되도록 한다.
cp_callback = make_checkpoint("training_super_small_3/cp-{epoch:04d}.ckpt")

# 학습률 계산을 위한 클래스를 불러온다.
learning_rate = CustomSchedule(D_MODEL)

# 옵티마이저 함수 선언
optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])


# 총 20번 훈련한다.
EPOCHS = 60
model.fit(dataset, epochs=EPOCHS, callbacks=[cp_callback])