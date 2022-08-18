TICKERS_FILE=$1

echo "Using tickers file: $TICKERS_FILE"

while read -r ticker; do
	echo "Processing $ticker";
	python main.py -en pnl-small-learning-rates\
	             -mind 2022-03-01\
               -maxd 2022-03-10 \
               -minde 2022-03-11 \
               -maxde 2022-03-11 \
               --ticker $ticker \
               -sgdi 20 \
               -sgdb 2048 \
               -rfl 1024 \
               -tbs 32768 \
               -sz 0.1 \
               -el 10 \
               -psr PnL \
               -nepw 6 \
               -epsr PnL \
               -wb True; # -mp "/home/data/best_model/checkpoint_000100/checkpoint-100" \

done < scripts/$TICKERS_FILE # assumes it is run from root of repo