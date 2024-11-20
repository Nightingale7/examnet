The model is adapted from
https://gitlab.com/gernst/ARCH-COMP/-/tree/FALS/models/FALS/neural

Based on observed outputs, it seems that the magnets position can go outside
[1, 3] which is the range for the reference point. It seems that it never goes
below 0 or above 4, so we use [0, 4] as the output range.
