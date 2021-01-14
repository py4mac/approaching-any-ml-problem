# Input file: https://raw.githubusercontent.com/TheClub4/IMDB_Sentiment_Analysis/master/movie_data.csv
INPUT_FILE = "./projects/imdb/inputs/imdb.csv"
TRAINING_FILE = "./projects/imdb/inputs/imdb_folds.csv"
MODEL_OUTPUT = "./projects/imdb/models/"
# https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
CRAWL_FILE = "./projects/imdb/inputs/crawl-300d-2M.vec"


MAX_LEN=128
TRAIN_BATCH_SIZE=16
VALID_BATCH_SIZE=8
EPOCHS=10