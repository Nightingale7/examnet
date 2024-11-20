The model file is from
https://github.com/ERATOMMSD/falstar/blob/master/falstar/resource/simulink/cars.mdl

Requirements described in https://easychair.org/publications/open/F4kf This
paper does not specify input ranges, but input ranges [0, 1] for both inputs
are given in "Falsification of hybrid systems using adaptive probabilistic
search" by Ernst et al. Since these are the authors of the FALSTAR tool who
have participated in ARCH-COMP many times, we take these ranges to be
authoritative enough. When THROTTLE is constantly 1 and BRAKE zero, then the
output variables take values is [-5000, 0], so we use output range [-5000, 0]
for all outputs.

