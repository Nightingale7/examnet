#!/bin/sh

python ./evaluate_ambiegen.py AMBIEGEN_EXAMNET ORG 12341 100 examnet
python ./evaluate_ambiegen.py AMBIEGEN_RANDOM ORG 12341 100 random
python ./evaluate_ambiegen.py AMBIEGEN_WOGAN ORG 12341 100 wogan
python ./evaluate_ambiegen.py AMBIEGEN_OGAN ORG 12341 100 ogan



