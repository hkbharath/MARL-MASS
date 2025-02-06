#!/bin/bash
sname==$(echo "$1" | awk -F'/' '{print $NF}')
old_seed=0
# screen -dmS "$sname-0" python run_mappo.py --config $1
# sleep 5

# rand_seeds=(2000 2024 123 4567 890 2743 1598 3621 490 0)
rand_seeds=(123 4567 890 2743 1598 3621 490 0)

for seed in "${rand_seeds[@]}"; do
    sleep 5
    sed -i "s/\bseed = $old_seed\b/seed = $seed/g" $1
    old_seed=$seed
    if [[ "$seed" -eq 0 ]]; then
        break
    fi
    screen -dmS "$sname-$seed" python run_mappo.py --config $1
done