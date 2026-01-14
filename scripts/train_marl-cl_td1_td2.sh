#!/bin/bash

# # # # # # # # # # # # # # # 
# Pure CVA Traffic
# # # # # # # # # # # # # # #

# Unsafe MARL - shared # 15800
checkpoint=15800
model_dirs=("/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_06_00_52_28" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_06_11_20_00" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_06_11_20_05" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_06_11_20_10" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_06_11_20_15")

# Unsafe MARL # 19600
checkpoint=19600
model_dirs=("/home/jovyan/work/cbf-avs_cint/Safe_MARL_CAVs/results/Jan_24_17_57_52" "/home/jovyan/work/cbf-avs_cint/Safe_MARL_CAVs/results/Jan_24_17_57_57" "/home/jovyan/work/cbf-avs_cint/Safe_MARL_CAVs/results/Jan_24_17_58_02" "/home/jovyan/work/cbf-avs_cint/Safe_MARL_CAVs/results/Feb_05_13_05_07" "/home/jovyan/work/cbf-avs_cint/Safe_MARL_CAVs/results/Feb_05_12_57_56")

# MARL-CMC # 20000
checkpoint=20000
model_dirs=("/home/jovyan/work/dmc/MARL-MASS/results/Sep_20_13_20_07" "/home/jovyan/work/dmc/MARL-MASS/results/Sep_20_13_20_18" "/home/jovyan/work/dmc/MARL-MASS/results/Sep_20_13_20_23" "/home/jovyan/work/dmc/MARL-MASS/results/Sep_20_13_20_28" "/home/jovyan/work/dmc/MARL-MASS/results/Sep_20_13_20_33")

# MARL-HSS # 11400
checkpoint=11400
model_dirs=("/home/jovyan/work/cbf-avs_cint/Safe_MARL_CAVs/results/Jan_24_17_56_11" "/home/jovyan/work/cbf-avs_cint/Safe_MARL_CAVs/results/Jan_24_17_56_16" "/home/jovyan/work/cbf-avs_cint/Safe_MARL_CAVs/results/Jan_24_17_56_21" "/home/jovyan/work/cbf-avs_cint/Safe_MARL_CAVs/results/Feb_04_14_39_18" "/home/jovyan/work/cbf-avs_cint/Safe_MARL_CAVs/results/Feb_04_14_39_38")

# # # # # # # # # # # # # # # 
# Mixed Traffic
# # # # # # # # # # # # # # #

# Unsafe MARL
# model_dirs=("/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Aug_25_17_01_34" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Aug_25_17_01_45" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Aug_25_17_01_50" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Aug_25_17_01_55" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Aug_25_17_02_00")

# Unsafe MARL - shared # 19200
# checkpoint=19200
# model_dirs=("/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_11_50" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_12_00" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_12_05" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_12_10" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_12_15")

# MARL-CMC
# model_dirs=("/home/jovyan/work/dmc/MARL-MASS/results/Sep_21_10_04_25" "/home/jovyan/work/dmc/MARL-MASS/results/Sep_21_10_04_35" "/home/jovyan/work/dmc/MARL-MASS/results/Sep_21_10_04_40" "/home/jovyan/work/dmc/MARL-MASS/results/Sep_21_10_04_45" "/home/jovyan/work/dmc/MARL-MASS/results/Sep_21_10_04_50")

# MARL-CMC - shared # 5200
# checkpoint=5200
# model_dirs=("/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_29_15_56_04" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_29_15_56_14" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_29_15_56_19" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_29_15_56_24" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_29_15_56_29")

# MARL-MASS
# model_dirs=("/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Aug_24_00_01_59" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Aug_24_00_02_09" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Aug_24_00_02_14" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Aug_24_00_02_19" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Aug_24_00_02_24")

# MARL-MASS - shared # 12800
# checkpoint=12800
# model_dirs=("/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_04_53" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_05_03" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_05_08" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_05_13" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_25_16_05_18")

# MARL-MASS - shared - cal # 5800
# checkpoint=5800
# model_dirs=("/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_29_16_06_44" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_29_16_06_54" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_29_16_06_59" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_29_16_07_04" "/home/jovyan/work/cbf-cav-mixed/MARL-MASS/results/Sep_29_16_07_09")

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