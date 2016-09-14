onlyeval=$1
model_name=$2
cp models/conll/* models/english/
python train_newdomain.py \
        --reload 1 --model $2 \
	--lr_method adam-lr_.001 --tag_scheme iob \
	--crf 0 --lower 1 --zero 1 \
	--train data/train_news.txt \
	--dev data/dev_news.txt \
	--test data/dev_news.txt \
	--onlyeval $onlyeval
