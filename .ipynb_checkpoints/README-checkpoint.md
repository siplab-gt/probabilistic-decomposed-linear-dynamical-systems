# Probabilistic Decomposed Linear Dynamical Systems

"Probabilistic Decomposed Linear Dynamical Systems for Robust Discovery of Latent Neural Dynamics."

Yenho Chen, Noga Mudrik, Adam Charles, Christopher J Rozell

This is the codebase for probabilistic decomposed linear dynamical systems (pdLDS). pdLDS improves robustness against temporal dynamics noise and system nonlinearity by extending the latent dynamics model in dLDS.  


# Installation
Try out installation from PyPi with the command

```
pip install pdLDS
```

### Install conda environment

If PyPi installation leads to problems with OMP (such as on Mac devices), it may be better to build directly from the provided conda environment.

```
git clone https://github.com/siplab-gt/probabilistic-decomposed-linear-dynamical-systems.git
cd probabilistic-decomposed-linear-dynamical-systems
conda env create -f environment.yml
conda activate pdLDS
```

TODO:
- Add citation
- Add tutorial
