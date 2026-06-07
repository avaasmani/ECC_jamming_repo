# ECC_jamming_repo

## Paper

*“Measuring Error-Control Code Performance in LEO Satellite Uplinks under Jamming”* submitted to ECE239AS (June 2026).  

**PDF:** `./paper_final.pdf`

**Contributors**

| Name        | Email                                             |
| ----------- | ------------------------------------------------- |
| Ava Asmani  | [](mailto:)   |
| Ting-Yu Yeh | [tingyu0225@g.ucla.edu](mailto:tingyu0225@g.ucla.edu)   |
| Thai Nguyen | [](mailto:) |

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
python simulation/run_ldpc.py
python simulation/run_polar.py
```

The generated results will be saved in:

```text
results/software/
```

---

## Reproducing SDR Experiments

The SDR experiments use over-the-air transmissions with:

* QPSK modulation
* Root-raised cosine pulse shaping
* Zadoff-Chu synchronization sequence
* Polar-coded header
* LDPC-coded payload

Example:

```bash
python sdr/transmitter.py
python sdr/receiver.py
```

Results will be stored in:

```text
results/sdr/
```






