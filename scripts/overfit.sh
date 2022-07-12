mind="2022-03-02"
maxd="2022-03-02"
minde="2022-03-02"
maxde="2022-03-02"
episode_length=60
rollout_fragment_length=600
train_batch_size=600
step_size=10
n_levels=50
fixed_ticker="JPM"
per_step_reward="PnL"
min_start_time="1000"
max_end_time="1100"


python main.py \
         -mind $mind \
         -maxd $maxd \
         -minde $minde \
         -maxde $maxde \
         -el $episode_length \
         -sz $step_size \
         -nl $n_levels \
         -t $fixed_ticker \
         -psr $per_step_reward \
         -rfl $rollout_fragment_length \
         -tb $train_batch_size \
         -min_st $min_start_time \
         -max_et $max_end_time