TICKERS_FILE='GE.txt'
#TICKERS_FILE='tickers_all.txt'
#TICKERS_FILE='scripts/tickers_batch1.txt'
#TICKERS_FILE='scripts/tickers_batch2.txt'
#TICKERS_FILE='scripts/single.txt'

mind="2018-02-20"
maxd="2018-02-20"
minde="2018-02-20"
maxde="2018-02-20"
episode_length=30

#mind="2022-03-01"
#maxd="2022-03-14"
#minde="2022-03-15"
#maxde="2022-03-30"
#episode_length=300
step_size=5
n_levels=50

#parallel python tune_rule_based_agents.py \
python tune_rule_based_agents.py \
         -mind $mind \
         -maxd $maxd \
         -minde $minde \
         -maxde $maxde \
         -el $episode_length \
         -sz $step_size \
         -nl $n_levels \
         -t GE \
         -psr PnL \
         #:::: $TICKERS_FILE \
         #::: PnL AD

