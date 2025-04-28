#!/bin/bash

theta=25
momentum=6
temperature=1.0
sampling="Default"
topK=3000
np=0.98
config_file="config/CA_config.json"

output_dir=$(python -c "
import json
with open('$config_file', 'r') as f:
    config = json.load(f)
print(config['Inference']['fixed_point_dir'])
")

if [ -z "$output_dir" ]; then
    echo "Error: Unable to extract output directory from config file."
    exit 1
fi

output_dir="Generations/$output_dir"

while [ $theta -le 155 ]
do
    if ls "${output_dir}"/*Pion*theta_${theta}* 1> /dev/null 2>&1; then
        echo "Pion file for theta $theta already exists. Skipping..."
    else
        python generate_fixedpoints.py --config "$config_file" --momentum $momentum --theta $theta --method "Pion" --temperature $temperature --sampling "$sampling" --topK $topK --nucleus_p $np
    fi

    if ls "${output_dir}"/*Kaon*theta_${theta}* 1> /dev/null 2>&1; then
       echo "Kaon file for theta $theta already exists. Skipping..."
    else
       python generate_fixedpoints.py --config "$config_file" --momentum $momentum --theta $theta --method "Kaon" --temperature $temperature --sampling "$sampling" --topK $topK --nucleus_p $np
    fi

    theta=$((theta + 5))
done

python make_plots.py --config "$config_file" --momentum $momentum
