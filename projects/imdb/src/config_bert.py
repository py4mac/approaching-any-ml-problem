import transformers

# Input file: https://raw.githubusercontent.com/TheClub4/IMDB_Sentiment_Analysis/master/movie_data.csv
INPUT_FILE = "./projects/imdb/inputs/imdb.csv"
TRAINING_FILE = "./projects/imdb/inputs/imdb_folds.csv"
MODEL_OUTPUT = "./projects/imdb/models/"
# https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
CRAWL_FILE = "./projects/imdb/inputs/crawl-300d-2M.vec"


MAX_LEN=512
TRAIN_BATCH_SIZE=8
VALID_BATCH_SIZE=4
EPOCHS=10

# https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
BERT_PATH = "./projects/imdb/inputs/bert_base_uncased/"

TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)