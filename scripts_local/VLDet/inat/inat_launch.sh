folder_path="scripts_CoDet_iNat/swin/"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done

folder_path="scripts_CoDet_iNat/rn50/"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done
