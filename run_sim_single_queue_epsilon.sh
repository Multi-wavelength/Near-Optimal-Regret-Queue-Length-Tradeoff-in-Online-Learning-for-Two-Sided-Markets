#!/bin/bash

# Script to reproduce results

for _epsilon_scaling in 0.6 0.8 1.0 1.2 1.4
do
    python main_single_queue_diff_para.py \
    --policy "learn_two_price_threshold" \
    --epsilon_scaling $_epsilon_scaling
done

