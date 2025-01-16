#!/bin/bash
sname==$(echo "$1" | awk -F'/' '{print $NF}')
screen -dmS "$sname-0" python run_mappo.py --config $1
sleep 5
sed -i 's/\bseed = 0\b/seed = 2000/g' $1
screen -dmS "$sname-2000" python run_mappo.py --config $1
sleep 5
sed -i 's/\bseed = 2000\b/seed = 2024/g' $1
screen -dmS "$sname-2024" python run_mappo.py --config $1
sleep 5
sed -i 's/\bseed = 2024\b/seed = 0/g' $1