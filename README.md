# Physics-Inspired Window-Wise Learning Framework for Capacity Degradation Prediction

Official companion repository for the manuscript:

**Physics-Inspired Window-Wise Learning Framework for Capacity Degradation Prediction using Diverse Real-World Battery Packs**

> **Paper status:** this manuscript is currently under review and has not been accepted for publication yet. The citation, DOI, and final archival links will be updated after the paper is accepted.

## Overview

This repository provides code and example data files for the proposed Physics-Inspired Window-Wise Learning (PIW2L) framework for battery capacity degradation prediction. The framework is designed for real-world battery packs and combines window-wise temporal learning with physics-inspired constraints for state-of-health/capacity prediction.

The current release is intended to support review, reproducibility checks, and future open-source maintenance. Only representative data samples are included in this repository; complete public datasets should be downloaded from their original sources.

## Data Availability

This work uses three real-world battery-pack datasets: **B20**, **VST**, and **LP**. To keep the repository lightweight and comply with data access restrictions, only sample files are provided.

| Dataset | Availability | Notes |
| --- | --- | --- |
| B20 | Public dataset | The dataset is publicly available at: <https://github.com/BatICM/battery-charging-data-of-on-road-electric-vehicles>. This repository includes one example file for format reference only. |
| VST | Public dataset | The dataset is publicly available at: <https://www.nature.com/articles/s41467-025-56485-7#Sec16>. Please obtain the complete data from the original source. |
| LP | Restricted enterprise dataset | The full LP dataset is confidential and cannot be redistributed. This repository only provides a partial sample for demonstrating the data format and processing workflow. |

When using the public datasets, please follow the data licenses, citation requirements, and usage terms specified by the original dataset providers.

## Environment

The code is developed in Python and uses PyTorch for model training.

Recommended dependencies:

```text
python >= 3.10
torch == 2.1.1+cu118
numpy >= 1.26.4
matplotlib >= 3.7.1
```

If you use CUDA, install the PyTorch build that matches your local CUDA version following the official PyTorch installation instructions.


## Citation

Because the manuscript is still under review, please do not cite this repository as an accepted publication. A BibTeX entry will be added after the paper is accepted.

```bibtex
@misc{piw2l_under_review,
  title  = {Physics-Inspired Window-Wise Learning Framework for Capacity Degradation Prediction using Diverse Real-World Battery Packs},
  author = {To be updated},
  note   = {Manuscript under review},
  year   = {2026}
}
```

## Notes

- This repository is released as companion code for an under-review manuscript.
- The included data files are examples and are not intended to replace the full public datasets.
- The LP dataset contains enterprise-confidential records; only a limited sample can be shared.
- Dataset paths and preprocessing details may be updated as the manuscript and code release are finalized.

## License

The project license will be finalized before the formal public release. Dataset licenses and access restrictions are governed by the original dataset providers.
