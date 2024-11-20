The model is adapted from
https://gitlab.com/gernst/ARCH-COMP/-/tree/FALS/models/FALS/steam-condenser

Based on observed outputs, the pressure output signal is initially lowest 86
and highest 90. Since the requirement is to have the pressure output signal
eventually between 87 and 87.5, it is safe to set its output range to [86, 90].
