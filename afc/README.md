The model is adapted from https://gitlab.com/gernst/ARCH-COMP/-/tree/FALS/models/FALS/powertrain

The model is based on the following paper:

X. Jin, J. V. Deshmukh, J. Kapinski, K. Ueda, and K. Butts.
Powertrain control verification benchmark.
In Proceedings of the 17th International Conference on Hybrid Systems: Computation and Control, HSCC ’14, p. 253–262, (2014).

As described in this paper, the output MU is a normalized error of a parameter
\lambda from a certain reference value. In principle MU can have arbitrarily
large absolute value, but as the thresholds \beta and \gamma in the
requirements are rather small, we found it reasonable to assume that MU takes
values in [0, 1]. The other ranges etc. come from the ARCH-COMP 2021 report.
