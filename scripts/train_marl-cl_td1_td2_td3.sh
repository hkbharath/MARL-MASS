#!/bin/bash


# Unsafe MARL # 16000
# checkpoint=16000
# model_dirs=("/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_55_50" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_55_55" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_56_00" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_56_05" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_56_10")

# Unsafe MARL - shared # 19000
# checkpoint=19000
# model_dirs=("/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Oct_01_12_14_35" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Oct_01_12_14_40" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Oct_01_12_14_45" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Oct_01_12_14_51" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Oct_01_12_14_55")

# MARL-CMC # 18400
# checkpoint=18400
# model_dirs=("/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Oct_01_11_02_31" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Oct_01_11_02_36" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Oct_01_11_02_41" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Oct_01_11_02_46" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Oct_01_11_02_51")

# MARL-CMC - shared # 9000
# checkpoint=9000
# model_dirs=("/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Oct_01_12_30_46" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Oct_01_12_30_51" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Oct_01_12_30_56" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Oct_01_12_31_01" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Oct_01_12_31_06")

# MARL-MASS # 8600
# checkpoint=8600
# model_dirs=("/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_39_35" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_39_40" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_39_45" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_39_50" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_39_55")

# MARL-MASS - shared # 12600
# checkpoint=12600
# model_dirs=("/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Oct_01_12_34_21" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Oct_01_12_34_26" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Oct_01_12_34_31" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Oct_01_12_34_36" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Oct_01_12_34_41")

# MARL-MASS - shared - cal # 20000
checkpoint=20000
model_dirs=("/home/jovyan/work/cbf-cav-mixed-cal/MARL-MASS/results/Oct_01_12_41_42" "/home/jovyan/work/cbf-cav-mixed-cal/MARL-MASS/results/Oct_01_12_41_47" "/home/jovyan/work/cbf-cav-mixed-cal/MARL-MASS/results/Oct_01_12_41_52" "/home/jovyan/work/cbf-cav-mixed-cal/MARL-MASS/results/Oct_01_12_41_57" "/home/jovyan/work/cbf-cav-mixed-cal/MARL-MASS/results/Oct_01_12_42_02")

sname=$(echo "$1" | awk -F'/' '{print $NF}')
old_seed=0
echo "Seed: $old_seed, Model:${model_dirs[0]}"
screen -dmS "$sname-0" python run_mappo.py --config $1 --model-dir ${model_dirs[0]} --checkpoint $checkpoint
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
    screen -dmS "$sname-$seed" python run_mappo.py --config $1 --model-dir $model --checkpoint $checkpoint
done