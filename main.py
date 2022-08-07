from utils import use_model, load_latest_checkpoint
from transformer import transformer
from nlp_utils import TextTokenizing
import argparse
from hyperparameters import NUM_LAYERS, D_MODEL, NUM_HEADS, DFF, DROPOUT


def main(args):
    checkpoint_dir = args.tokenizer
    latest_checkpoint = load_latest_checkpoint(checkpoint_dir)

    textTokenizing = TextTokenizing()

    tokenizer = textTokenizing.load_tokenizer(args.tokenizer)

    VOCAB_SIZE, START_TOKEN, END_TOKEN = textTokenizing.tokens()


    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        dff=DFF,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    )

    prediction = use_model(model=model, tokenizer=tokenizer, START_TOKEN=START_TOKEN, END_TOKEN=END_TOKEN, MAX_LENGTH=50)
    
    while True:
        user_input = input("You: ")

        machine_answer = prediction.predict(user_input)

        print(f"AI: {machine_answer}")



if __name__ == '__main__':
    """Arguments Parser"""
    argparser = argparse.ArgumentParser(description="Human Like Chatbot Training")


    argparser.add_argument(
        '--tokenizer',
        default="tokenizer",
        help="Directory of the .subword tokenized words file."
    )

    argparser.add_argument(
        '--checkpoint',
        default="training",
        help="Directory where checkpoint file was stored"
    )


    
    args = argparser.parse_args()

    main(args)