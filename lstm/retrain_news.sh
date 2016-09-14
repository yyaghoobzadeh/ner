python train_newdomain.py --reload 1 --model english \
	--lr_method adam-lr_.0001 --tag_scheme iob \
	--crf 0 --lower 1 --zero 1 \
	--train data/train_news.txt \
	--dev data/dev_news.txt \
	--test data/dev_news.txt \
