echo $1
#train the LSTM CHAR DROPOUT VAE
python train.py --lr_method adam-lr_.001 --lower 1 --zeros 1 --char_dim 5 \
	--char_lstm_dim 10 \
	--char_bidirect 0 --word_dim 50 \
	--word_lstm_dim 50 --word_bidirect 1 \
	--cap_dim 0 --crf 0 --dropout 0.5 \
	--tag_scheme iob \
	--pre_emb glove.6B.50d.txt --all_emb 0 \
       	--train eng.train --dev eng.testa --test eng.testb \
	--model $1 \
	--onlyeval 0





