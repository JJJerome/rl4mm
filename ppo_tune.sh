#TICKERS_FILE='scripts/tickers_batch1.txt'
#TICKERS_FILE='scripts/tickers_batch2.txt'
TICKERS_FILE='scripts/single.txt'
#TICKERS_FILE='tickers_all.txt'

# Overfitting test:
mind="2022-03-01"
maxd="2022-03-11" # 11th was a Friday.
# Eval on the week that followed:
minde="2022-03-14"
maxde="2022-03-18" 
episode_length=210
step_size=1
n_levels=50
concentration=10.0
min_st="1100"
max_et="1430"

parallel echo python main.py \
         -mind $mind \
         -maxd $maxd \
         -minde $minde \
         -maxde $maxde \
	 -min_st $min_st \
         -max_et $max_et \
         -nl $n_levels \
         -sz $step_size \
         -c  $concentration \
         -el $episode_length \
         -t {1} \
         -psr {2} \
	 -n {3} \
         :::: $TICKERS_FILE \
         ::: PnL AD \
         ::: True False

