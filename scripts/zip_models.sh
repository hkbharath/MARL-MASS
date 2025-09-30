
model_folders_td1=("Sep_21_10_04_50" "Sep_21_10_04_45" "Sep_21_10_04_40" "Sep_21_10_04_35" "Sep_21_10_04_25")

model_folders_td2=("Sep_21_08_42_09" "Sep_21_08_42_04" "Sep_21_08_41_59" "Sep_21_08_41_54" "Sep_21_08_41_44")

model_folders_td3=("Sep_21_08_38_45" "Sep_21_08_38_40" "Sep_21_08_38_35" "Sep_21_08_38_30" "Sep_21_08_38_21")

for model_folder in "${model_folders_td1[@]}"; do
    zip -r "$model_folder.zip" "$model_folder"
done

for model_folder in "${model_folders_td2[@]}"; do
    zip -r "$model_folder.zip" "$model_folder"
done

for model_folder in "${model_folders_td3[@]}"; do
    zip -r "$model_folder.zip" "$model_folder"
done
