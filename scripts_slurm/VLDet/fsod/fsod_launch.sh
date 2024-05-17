folder_path="scripts_CoDet_FSOD/swin/"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done

folder_path="scripts_CoDet_FSOD/rn50/"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done
