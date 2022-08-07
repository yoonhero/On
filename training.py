from ast import arg
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from nlp_utils import preprocess_sentence, TextTokenizing
from transformer import transformer, CustomSchedule, loss_function
from utils import LoggingResult, make_checkpoint, accuracy, load_csv_and_processing
from hyperparameters import NUM_LAYERS, D_MODEL, NUM_HEADS, DFF, DROPOUT
import argparse



@LoggingResult
def training_model(args):
    # Loading DataSet
    # CSV Structure Example
    # Q                |                 A 
    # 뭐해?            |               코딩!       
    questions, answers = load_csv_and_processing(args.dataset)


    # 토크나이저 모듈 
    textTokenizing = TextTokenizing()

    if args.create_tokenizer:
        # 토크나이저를 생성한다.
        _ = textTokenizing.create_tokenizer(questions, answers, target_vocab_size=args.target_vocab_size)

        # 토크나이저를 저장해서 다음 실행 때 시간을 단축하고 싶다면 사용
        textTokenizing.save_tokenizer(args.tokenizer)

    # 토크나이저를 로딩
    textTokenizing.load_tokenizer(args.tokenizer)


    # 보캡 사이즈, 스타트 토큰, 엔드 토큰을 불러온다.
    VOCAB_SIZE, START_TOKEN, END_TOKEN = textTokenizing.tokens()

    print(f"보캡 사이즈: {VOCAB_SIZE}, 스타트 토큰: {START_TOKEN}, 엔드 토큰: {END_TOKEN}")


    # 입력된 문장을 토크나이징한다.
    questions, answers = textTokenizing.tokenize_and_filter(questions, answers)


    print(f'질문 데이터의 크기:{questions.shape}')
    print(f'답변 데이터의 크기:{answers.shape}')


    # 배치 크기와 버퍼 사이즈를 선언
    BATCH_SIZE = args.batch
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
    cp_callback = make_checkpoint(args.checkpoint, args.save_best_only)

    # 학습률 계산을 위한 클래스를 불러온다.
    learning_rate = CustomSchedule(D_MODEL)

    # 옵티마이저 함수 선언
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])


    # 총 20번 훈련한다.
    EPOCHS = args.epochs
    model.fit(dataset, epochs=EPOCHS, callbacks=[cp_callback])




def main():
    """Arguments Parser"""

    argparser = argparse.ArgumentParser(description="Human Like Chatbot Training")

    argparser.add_argument(
        '-v', '--verbose',dest="debug", action="store_true", dest="debug", help="Print debug information"
        )

    argparser.add_argument(
        '--dataset',
        default="./dataset.csv",
        help="Directory of the dataset csv file (default: ./dataset.csv)"
    )

    argparser.add_argument(
        '--tokenizer',
        default="tokenizer",
        help="Directory of the .subword tokenized words file."
    )

    argparser.add_argument(
        '--craete-tokenizer',
        dest="create_tokenizer",
        action="store_false",
        type="bool",
        help="Create Tokenized Words File"
    )

    argparser.add_argument(
        '--target-vocab-size',
        dest="target_vocab_size",
        type="int",
        default=2**15,
        help="Set a Vocab Size"
    )

    argparser.add_argument(
        '--checkpoint',
        default="training/cp-{epoch:04d}.ckpt",
        help="Directory where checkpoint file will be stored"
    )

    argparser.add_argument(
        '--batch-size',
        dest="batch",
        type="int",
        default=64,
        help="Set Batch Size"
    )
    

    argparser.add_argument(
        '--save-best-only',
        type="bool",
        action="store_false"
        dest="save_best_only",
        help="Save The Best Model or not"
    )

    argparser.add_argument(
        '--epochs', 
        type="int"
        default="20"
        help="Set Epoch of Training Process"
    )

    
    args = argparser.parse_args()

    training_model(args)



if __name__ == "__main__":
    main()