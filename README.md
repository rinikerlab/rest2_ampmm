# REST2-AMP/MM

![Logo](AMP_logo.png)

## Overview

**REST2-AMP/MM** provides the code, parameters, and pretrained weights for the  
**AMPv3-BMS25 multiscale neural network potential**, together with full support for  
**AMP/MM** and **REST2-AMP/MM** simulation workflows.

This repository enables:

- Multiscale neural network simulations of proteins, peptides, and small organic molecules  
- **AMP/MM equilibration and production MD runs**  
- **REST2-AMP/MM enhanced sampling simulations**

---

## Installation

This project is designed to be used with **conda/mamba**.

### 1. Create environment

```bash
conda env create -f environment.yaml
conda activate rest2_ampmm
```

### 2. Install the package

```bash
pip install -e .
```

---

## Model Description

AMP-BMS is a **multiscale neural network potential** trained on QM/MM data with electrostatic embedding for simulations of biomolecules in the condensed phase.

The model:

- is based on the AMP architecture  
- is trained on high-level QM/MM reference data  
- enables efficient and accurate condensed-phase simulations  

### Related Work

- REST2-AMP/MM: **[Placeholder for publication]**  
- AMPv3-BMS25: https://chemrxiv.org/doi/full/10.26434/chemrxiv-2026-kx9w0  
- BMS25 dataset: https://www.research-collection.ethz.ch/entities/researchdata/15515ac0-d9a6-4966-b658-5e391907ef43  
- BMS25 publication: https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-xzp6k  
- AMPv2: https://pubs.acs.org/doi/full/10.1021/jacs.4c17015  
- AMP framework: https://openreview.net/forum?id=socffUzSIlx  
- ANA2B: https://pubs.rsc.org/en/content/articlehtml/2023/sc/d3sc04317g  

---

## Running Simulations

Simulations are typically carried out through the following stages:

```
NVT → NPT → production or REST2
```

Each stage consists of:
1. Generating a `config.yaml`
2. Submitting it to the scheduler

---

### General Setup

Before running any stage, generate input files using the corresponding script.

The molecule name determines the QM region and charge and must be consistent across all steps.

**Placeholders:**

- `<PDB_NAME>` — system name (e.g. `ALA`)  
- `<QM_RESIDUES>` — QM region (e.g. `ACE ALA NME`)  
- `HOH` — MM region (water)  
- `<CHARGE>` — QM charge (e.g. `0` for neutral systems)  

Ensure consistency across all scripts:

```
nvt.py, npt.py, production.py, rest2.py
```

---

### Stage Commands

#### NVT

```bash
python input_generation_scripts_examples/nvt.py \
  --molecule <PDB_NAME> \
  --qm_zone <QM_RESIDUES> \
  --mm_zone HOH \
  --mol_charge <CHARGE> \
  --output nvt
```

#### NPT

```bash
python input_generation_scripts_examples/npt.py \
  --molecule <PDB_NAME> \
  --qm_zone <QM_RESIDUES> \
  --mm_zone HOH \
  --mol_charge <CHARGE> \
  --input nvt \
  --output npt
```

#### Production

```bash
python input_generation_scripts_examples/production.py \
  --molecule <PDB_NAME> \
  --qm_zone <QM_RESIDUES> \
  --mm_zone HOH \
  --mol_charge <CHARGE> \
  --restart_velocities  # if needed \
  --input npt \
  --output production_0000
```

#### REST2

Define:
- number of replicas  
- scaling range  

Scaling distributions are configurable in:
```
src/helper_functions.py
```

```bash
python input_generation_scripts_examples/rest2.py \
  --molecule <PDB_NAME> \
  --qm_zone <QM_RESIDUES> \
  --mm_zone HOH \
  --mol_charge <CHARGE> \
  --restart_velocities  # if needed \
  --input npt \
  --output rest2_0000_0000
```

---

### Output Structure

Each stage creates:

```
amp_simulation/<PDB_NAME>/<STAGE>/
```

containing:
- `config.yaml` (editable)

---

### Submitting Jobs

```bash
sbatch --wrap="rest2-ampmm amp_simulation/<PDB_NAME>/<STAGE>/config.yaml"
```

- `<STAGE>`: `nvt`, `npt`, `production`, `rest2`

Ensure SLURM resources are compatible with your simulation setup.

---

## Automated SLURM Submission

Example script:

```
submission_scripts_examples/submit_SLURM_simulations.sh
```

This script automates:
- input generation  
- job submission  

For **production** and **REST2**, it submits a chain of dependent jobs:
- one job per trajectory segment  
- sequential execution (`afterok`)  
- automatic restart from previous segment  

---

### Usage

```bash
bash submission_scripts_examples/submit_SLURM_simulations.sh <MOLECULE> <STAGE> [N_REPS]
```

- `<MOLECULE>`:  
  `ALA`, `CYC2`, `CYC2t`, `P13n`, `P13e`, `TRP`, `TRPu`, `CLN`, `CLNunf`, `CHICF`, `CHIC`  
- `<STAGE>`: `nvt`, `npt`, `production`, `rest2`  
- `[N_REPS]`: number of **independent replicates** (default = 1)

> **Note**  
> `N_REPS` refers to independent simulations (replicates),  
> not to replicas used internally in REST2.

---

### Examples

```bash
# NVT equilibration (single job)
bash submission_scripts_examples/submit_SLURM_simulations.sh ALA nvt

# NPT equilibration (single job, continues from NVT)
bash submission_scripts_examples/submit_SLURM_simulations.sh ALA npt

# Production (chain of jobs, 1 independent replicate)
bash submission_scripts_examples/submit_SLURM_simulations.sh ALA production 1

# REST2 (MPI + GPU, 1 independent replicate)
bash submission_scripts_examples/submit_SLURM_simulations.sh ALA rest2 1
```

---

### Notes

- **NVT / NPT**  
  - single SLURM job  

- **Production / REST2**  
  - segmented trajectories  
  - sequential job dependencies  
  - support multiple independent replicates  

- Logs are written to:
```
logs/
```

---
## Repository Structure

### `input_generation_script_examples/`

Contains examples for generating YAML configuration files for:

- NVT equilibration (`nvt.py`)
- NPT equilibration  (`npt.py`)
- Production MD  (`production.py`)
- REST2-AMP/MM simulations  (`rest2.py`)

---

### `submission_script_examples/`

Contains example scripts for **HPC submission of chained REST2-AMP/MM and production MD AMP/MM jobs**, where:

- MD simulations are split into chunks  
- Each chunk is submitted as a dependency of the previous job
- The jobs are run on a cluster with SLURM 

---

### `analysis/`

Contains analyses related to the REST2-AMP/MM publication, including:

- NOE violation calculations (`cyclic_peptides.py`, `trp_noe.py`, `cln_noe_folded.py`, `cln_noe_unfolded.py`)
- Roundtrips and exchange analysis (`roundtrips_exchange_acceptance.py`)
- C-alpha RMSD calculations (`trp_structural_analysis.py`, `cln_chignolin.py`) 
- RMSD of interatomic distances (`cln_chignolin.py`)
- Ramachandran analysis (`alanine_polyproline.py`, `cyclic_peptides.py`)
---

### `data/`

Contains:
- Starting structures of all systems in PDB format related to the REST2-AMP/MM publication (`starting_structures`)
- Examples of config.yaml files used for NVT equilibration, NPT equilibration, production MD, and REST2-AMP/MM runs (`example_input_configs`)

related to the REST2-AMP/MM publication

### `configs/deafult/`

Contains:
- Default JSON configurations of OpenMM/MDTraj StateDataReporter (`default_csv_parameters.json`) and HDF5 reporter (`default_hdf5_parameters.json`)
- Default JSON configurations of OpenMM force field XML files to be used with TIP4P-FB (`ff_default_tip4pfb.json`) and TIP3P (`ff_default.json`) water models
---

### `resources/`

Contains:
- AMPv3 YAML parameter files to be used with TIP4P-FB (`PARAMETERS_MIN_LRv2_tip4pfb.yaml`) and TIP3P water models (`PARAMETERS_MIN_LRv2.yaml`)
- AMPv3-BMS25 models weights (`MIN_LRv2_state_dict`)
---

## References

If you use this code, please cite our papers:

```

@article{REST2_AMP,
    title = {Quantum-Accurate Enhanced Sampling and Mini-Protein Folding through Replica Exchange for Multiscale Neural Network Potentials},
    author = {Solazzo, Riccardo and Gordiy, Igor and Riniker, Sereina},
    journal = {ChemRxiv},
    year = {2026}
    doi = {\url{PLACEHOLDER}},

}

@article{AMP_BMS,
    title = {{AMP}-{BMS}/{MM}: A Multiscale Neural Network Potential for the Fast and Accurate Simulation of Protein Dynamics and Enzymatic Reactions},
    author = {Th{\"u}rlemann, Moritz and Pultar, Felix and Gordiy, Igor and Ruijsenaars, Enrico and Riniker, Sereina},
    journal = {ChemRxiv},
    year = {2026}
    doi = {\url{https://chemrxiv.org/doi/10.26434/chemrxiv-2026-kx9w0}},
}

@article{BMS25,
title = {Biomolecular Simulation ({BMS}) Dataset to Train Multiscale Neural Network Potentials},
author = {Th{\"u}rlemann, Moritz and Pultar, Felix and Gordiy, Igor and Riniker, Sereina},
journal = {ChemRxiv},
year = {2025},
doi = {\url{https://chemrxiv.org/doi/10.26434/chemrxiv-2025-xzp6k}},
}

@article{AMPv2,
title = {Neural Network Potential with Multiresolution Approach Enables Accurate Prediction of Reaction Free Energies in Solution},
author = {Pultar, Felix and Th{\"u}rlemann, Moritz and Gordiy, Igor and Doloszeski, Eva and Riniker, Sereina},
journal = {J. Am. Chem. Soc.},
volume = {147},
pages = {6835--6856},
year = {2025},
doi={https://doi.org/10.1021/jacs.4c17015}
}


@article{AMP,
  title = {Anisotropic Message Passing: Graph Neural Networks with Directional and Long-Range Interactions},
  author = {Th{\"u}rlemann, Moritz and Riniker, Sereina},
  journal = {The Eleventh International Conference on Learning Representations},
  year = {2023},
  doi = {\url{https://openreview.net/forum?id=socffUzSIlx}},
  note = {(accessed 2025-12-13)},
}

```

## Contributors
- Igor Gordiy
- Riccardo Solazzo
- Moritz Thürlemann 
- Felix Pultar
- Enrico Ruijsenaars

## Acknowledgements
The authors thanks Antonia Kuhn and Stefan Chrin for code review and Francisco J. Enguita for the logo design. 

## License

The AMP-BMS code is published and distributed under the [MIT License](LICENSE). 