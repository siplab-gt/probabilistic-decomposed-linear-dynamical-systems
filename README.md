# Probabilistic Decomposed Linear Dynamical Systems

"Probabilistic Decomposed Linear Dynamical Systems for Robust Discovery of Latent Neural Dynamics." (NeurIPS 2024)

Yenho Chen, Noga Mudrik, Kyle A. Johnsen, Sankaraleengam Alagapan, Adam S. Charles, Christopher J. Rozell

[[Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bcf26768143c94bd36e363cd4bf5daf0-Abstract-Conference.html)] [[arXiv](https://arxiv.org/abs/2408.16862)] 

This is the codebase for probabilistic decomposed linear dynamical systems (pdLDS). pdLDS improves robustness against temporal dynamics noise and system nonlinearity by extending the latent dynamics model in dLDS.  


## Installation

**From PyPI:**
```bash
pip install pdLDS
```

### Install conda environment

**From source (recommended if PyPI install fails, e.g. OMP issues on macOS):**
```bash
git clone https://github.com/siplab-gt/probabilistic-decomposed-linear-dynamical-systems.git
cd probabilistic-decomposed-linear-dynamical-systems
conda env create -f environment.yml
conda activate pdLDS
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{chen2024probabilistic,
  title={Probabilistic decomposed linear dynamical systems for robust discovery of latent neural dynamics},
  author={Chen, Yenho and Mudrik, Noga and Johnsen, Kyle A and Alagapan, Sankaraleengam and Charles, Adam S and Rozell, Christopher J},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={104443--104470},
  year={2024}
}
```
