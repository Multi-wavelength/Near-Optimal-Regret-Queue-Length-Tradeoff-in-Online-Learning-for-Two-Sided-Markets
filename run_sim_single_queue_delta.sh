#!/bin/bash

# Script to reproduce results

for _delta_scaling in 0.1 0.2 0.3 0.4 0.5
do
    python main_single_queue_diff_para.py \
    --policy "learn_two_price_threshold" \
    --delta_scaling $_delta_scaling
done

