import tokenizers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 10
BERT_PATH = "../app/input/bert-base-uncased"
MODEL_PATH = "../app/models/model.bin"
TRAINING_FILE = "../app/input/train.csv"
TEST_FILE = "../app/input/test.csv"

print('bert', f"{BERT_PATH}/vocab.txt")
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    f"{BERT_PATH}/vocab.txt",
    lowercase=True
)
