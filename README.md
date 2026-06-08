# ECC_jamming_repo

## Paper

*“Measuring Error-Control Code Performance in LEO Satellite Uplinks under Jamming”* submitted to ECE239AS (June 2026).  

**PDF:** `./ECC_Jamming.pdf`

**Contributors**

| Name        | Email                                             |
| ----------- | ------------------------------------------------- |
| Ava Asmani  | [ava24@g.ucla.edu](mailto:ava24@g.ucla.edu)   |
| Ting-Yu Yeh | [tingyu0225@g.ucla.edu](mailto:tingyu0225@g.ucla.edu)   |
| Thai Nguyen | [thainguyen16@g.ucla.edu](mailto:thainguyen16@g.ucla.edu) |

---

## Project Overview

This project investigates the performance of LDPC and Polar error-control codes in LEO satellite uplink communications under jamming conditions. Both software simulations and software-defined radio (SDR) experiments were conducted to evaluate communication reliability using frame error rate (FER) and bit error rate (BER) measurements.

---

## Directory Guide

| Path            | Purpose                                                  |
|-----------------|----------------------------------------------------------|
| `scripts/`      | Scripts used to run simulations and SDR experiments.     |
| `figures/`      | Figures used in the paper and presentation.              |
| `figure_data/`  | Data generated from simulations and SDR experiments used to create the figures. |

---

## Reproducing Software Simulations

Run the simulation scripts to generate FER results under different jammer-to-noise ratios (JNR):

```bash
python3 scripts/fig2_ldpcfer.py
python3 scripts/fig3_polarawgn.py
python3 scripts/fig4_awgn_vs_rayleigh.py
```

The generated results will be saved in:

```text
figures/fig2_ldpcfer.png
figures/fig3_polarawgn.png
figures/fig4_awgn_vs_rayleigh.png
```

---

## Reproducing SDR Experiments

Run simulation with
```bash
python3 scripts/fig5_fig6_sdr_fer_ber.py
```
Note: you must have access to the Robert's Lab Remote RF platform. Once permission for SDR use has been granted, connectivity and installation info can be found here: https://wireless.ee.ucla.edu/remoterf/
```bash
pip install remoterf
```

Results will be stored in:

```text
figures/fig5_sdrfer.png
figures/fig6_sdrber.png
```






