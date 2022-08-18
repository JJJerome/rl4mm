TICKERS_FILE=$1

echo "Using tickers file: $TICKERS_FILE"

while read -r ticker; do
	echo "Processing $ticker";
	python main.py -en pnl-small-learning-rates \
	             -mind 2022-03-01\
               -maxd 2022-03-10 \
               -minde 2022-03-11 \
               -maxde 2022-03-11 \
               --ticker $ticker \
               -mp /home/RL4MM/ray_results/tensorboard/JPM/PnL/concentration_None/full_state/normalisation_on_False/moc_False/PPO/PPO_HistoricalOrderbookEnvironment_e6821_00007_7_lr=0.0001_2022-08-17_21-05-41/checkpoint_000004 \
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