#TICKERS_FILE='tickers_all.txt'
TICKERS_FILE='tickers_some.txt'

mind="2022-03-01"
maxd="2022-03-14"

while read -r ticker; do 
	echo "Processing $ticker";
	python rule_based_main.py -mind $mind -maxd $maxd --ticker $ticker;
done < scripts/$TICKERS_FILE # assumes it is run from root of repo
