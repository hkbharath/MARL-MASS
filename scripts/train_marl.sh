#!/bin/bash
screen python run_mappo.py --config $1
sed -i 's/\bseed = 0\b/seed = 2000/g' $1
screen python run_mappo.py --config $1
sed -i 's/\bseed = 2000\b/seed = 2024/g' $1
screen python run_mappo.py --config $1
sed -i 's/\bseed = 2024\b/seed = 0/g' $1