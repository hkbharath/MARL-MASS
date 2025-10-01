#!/bin/bash


# Unsafe MARL
# model_dirs=("/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Aug_25_17_01_34" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Aug_25_17_01_45" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Aug_25_17_01_50" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Aug_25_17_01_55" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Aug_25_17_02_00")

# Unsafe MARL - shared
model_dirs=("/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_11_50" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_12_00" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_12_05" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_12_10" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_12_15")

# MARL-CMC
# model_dirs=("/home/jovyan/work/dmc/MARL-MASS/results/Sep_21_10_04_25" "/home/jovyan/work/dmc/MARL-MASS/results/Sep_21_10_04_35" "/home/jovyan/work/dmc/MARL-MASS/results/Sep_21_10_04_40" "/home/jovyan/work/dmc/MARL-MASS/results/Sep_21_10_04_45" "/home/jovyan/work/dmc/MARL-MASS/results/Sep_21_10_04_50")

# MARL-MASS
# model_dirs=("/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Aug_24_00_01_59" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Aug_24_00_02_09" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Aug_24_00_02_14" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Aug_24_00_02_19" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Aug_24_00_02_24")

sname=$(echo "$1" | awk -F'/' '{print $NF}')
old_seed=0
echo "Seed: $old_seed, Model:${model_dirs[0]}"
screen -dmS "$sname-0" python run_mappo.py --config $1 --model-dir ${model_dirs[0]} --checkpoint $2
# sleep 5

rand_seeds_3=(2000 2024 0)
rand_seeds_5=(2000 2024 123 1598 0)
rand_seeds_10=(2000 2024 123 4567 890 2743 1598 3621 490 0)


for i in "${!rand_seeds_5[@]}"; do
    seed=${rand_seeds_5[$i]}
    model=${model_dirs[$((i+1))]}
    sleep 5
    echo "Seed: $seed, Model:$model"
    sed -i "s/\bseed = $old_seed\b/seed = $seed/g" $1
    old_seed=$seed
    if [[ "$seed" -eq 0 ]]; then
        break
    fi
    screen -dmS "$sname-$seed" python run_mappo.py --config $1 --model-dir $model --checkpoint $2
done