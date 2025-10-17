#!/bin/bash

# Script to reproduce results

for i in 5 6 7 8 9 10
do
    python main_single_queue.py \
    --policy "learn_two_price_threshold" \
    --order_alpha $(awk "BEGIN {print $i/120}")
done

