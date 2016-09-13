python train.py --train data/train_news.txt --dev data/dev_news.txt --test data/test_news.txt \
--dev data/dev_news.txt \
--model NoW2vCh5Lstm --tag_scheme iob \
--test data/test_news.txt \
--lower 1 --zeros 1 --char_dim 5 --char_lstm_dim 10 \
--char_bidirect 0 --word_dim 50 \
--word_lstm_dim 50 --word_bidirect 1 \
 --cap_dim 0 --crf 0 --dropout 0.2 --lr_method adam-lr_.001
