mind="2022-03-01"
maxd="2022-03-14"
minde="2022-03-15"
maxde="2022-03-18"
episode_length=60
step_size=10
n_levels=50
fixed_ticker="JPM"
per_step_reward="PnL"

python tune_rule_based_agents.py \
         -mind $mind \
         -maxd $maxd \
         -minde $minde \
         -maxde $maxde \
         -el $episode_length \
         -sz $step_size \
         -nl $n_levels \
         -t $fixed_ticker \
         -psr $per_step_reward