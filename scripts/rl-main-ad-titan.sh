TICKERS_FILE=$1

echo "Using tickers file: $TICKERS_FILE"

while read -r ticker; do
	echo "Processing $ticker";
	python main.py -mind 2022-03-01\
               -maxd 2022-03-10\
               -minde 2022-03-11\
               -maxde 2022-03-11\
               --ticker $ticker\
               -el 5\
               -psr AD\
               -epsr AD\
	       --num_workers 8\
	       --num_workers_eval 1\\
	       --num_gpus 1;

done < scripts/$TICKERS_FILE # assumes it is run from root of repo
