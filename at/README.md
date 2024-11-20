The model is from
https://gitlab.com/gernst/ARCH-COMP/-/tree/FALS/models/FALS/transmission

When THROTTLE is constantly 100 and BRAKE zero, then the top SPEED and top RPM
achieved during 30 time units are respectively 120.21327149261522 and
4768.277187862124. Thus we use output ranges [0, 121] and [0, 4800]. There are
four gears, so the output range for GEAR is [0, 4]. The other ranges come from
the ARCH-COMP 2021 report.
