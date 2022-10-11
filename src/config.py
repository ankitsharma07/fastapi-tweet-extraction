import tokenizers


MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 10
BERT_PATH = "../input/bert-base-uncased/"
MODEL_PATH = "../models/model.bin"
TRAINING_FILE = "../input/train.csv"
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    f"{BERT_PATH}/vocab.txt",
    lowercase=True
)
