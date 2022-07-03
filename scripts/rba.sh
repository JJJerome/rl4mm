#TICKERS_FILE='tickers_all.txt'
#TICKERS_FILE='tickers_some.txt'

TICKERS_FILE=$1

echo "Using tickers file: $TICKERS_FILE"

mind="2022-03-01"
maxd="2022-03-14"
n_iterations=60
episode_length=60
step_size=5
n_levels=50

while read -r ticker; do 
	echo "Processing $ticker";
	python rule_based_main.py -par -mind $mind -maxd $maxd -t $ticker -ni $n_iterations -el $episode_length -sz $step_size -nl $n_levels;
done < scripts/$TICKERS_FILE # assumes it is run from root of repo
