The Matlab model is from
https://gitlab.com/gernst/ARCH-COMP/-/tree/FALS/2019/FALS/falstar/models as the
code in https://gitlab.com/gernst/ARCH-COMP/-/tree/FALS/models/FALS/f16-gcas is
broken.

The Python versions are based on branches v1 (Python 2 version) and v2 (Python
3 version) of https://github.com/stanleybak/AeroBenchVVPython/ They seem to
behave differently as minimum achievable altitudes from initial altitude 4040
seem to be approximately 81 (Python 2) and 1697 (Python 3). These Python
versions differ from the Matlab model because the Matlab model is falsifiable
from initial altitude 4040.

In the paper "Part-X: A family of stochastic algorithms for search-based test
generation with probabilistic guarantees", a F16 model is used which the
authors claim to be unfalsifiable from initial altitude 2400. Since all of the
above models can be falsified from this altitude, it seems that this paper uses
yet another different model.

