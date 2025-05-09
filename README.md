# [Foundation Models for Imaging Cherenkov Detectors]()

# Abstract

Foundation models ....

![Example Hit Patterns](assets/Overlayed_hits.png)

# Contents
- [Data used for Training](#Section-1)
- [Environment](#Section-1)
- [Available Architectures](#Section-3)
- [Usage](#Section-4)
- [Reference Packages](#Section-5)


# Data used for Training

The data used for training our generative models was created using a standalone simulation, and is the same as used in [A Fast Simulation Suite for Cherenkov Detectors at the future Electron Ion Collider](https://arxiv.org/abs/2504.19042). More details can be found at [eicdirc](https://github.com/rdom/eicdirc).
Our dataset was constructed over a singular bar (no azimuthal angle variance), and without magnetic field. 

# Environment 

Noteable requirements: 

- Python:     3.12.8
- Pytorch:    2.5.1
- CUDA:       12.4

The dependencies for the networks can be installed with the following command:

```bash
conda env create -f env.yml
```

In the case that some packages do not install through the provided conda command, you can install them using pip once your conda environment is activated:

```bash
python3 -m pip install <package>
```


# Architecture

Insert image with architecture...

# Usage 

Note we have provided all code required to reproduce the results found within the paper, although these require large amounts of simulation from GEANT4. For those interested in training their own models, or reproducing our work, please open an issue. If their exists a high demand, we will update our documentation to and provide instructions for dataset creation, model training and model evaluation. We also plan to release a streamlined, containerized version for those who only wish to utilize the fast simulation methods for dataset creation.

## Running Fast Simulation - NEED TO DO

We have provided an example script to allow simulation at both fixed kinematics, and continuously over the phase space. An example command and argument details is provided below:

```
python run_simulation.py --config config/config.json --n_tracks {} --n_dump {} --method {} --momentum {} --theta {} --model_type {} --fine_grained_prior --dark_rate --dark_noise 22800
```

| Argument               | Type    | Default       | Description                                                              |
|------------------------|---------|---------------|--------------------------------------------------------------------------|
| `--config`             | `str`   | `config.json` | Path to the config file                                                  |
| `--n_tracks`           | `int`   | `1e5`         | Number of particles to generate                                          |
| `--n_dump`             | `int`   | `None`        | Number of particles to dump per `.pkl` file                              |
| `--method`             | `str`   | `"MixPK"`     | Generated particle type (`Kaon`, `Pion`, or `MixPK`)                     |
| `--momentum`           | `str`   | `"3"`         | Momentum to generate     (e.g., "6", "1-10")                             |
| `--theta`              | `str`   | `"30"`        | Theta angle to generate  (e.g., "30", "25-155")                          |
| `--model_type`         | `str`   | `"NF"`        | Which model to use (`NF`,`CNF`,`FlowMatching`,`Score`,`DDPM`)            |
| `--fine_grained_prior` | `flag`  | `False`       | Enable fine-grained prior (just include the flag to activate the option) |
| `--dark_noise`         | `flag`  | `False`       | Whether or not to include dark rate in generations                       |
| `--dark_rate`          | `float` | `22800`       | Dark rate - default corresponds to -c 2031 in standalone Geant simulation|

## Hit Pattern Creation - NEED TO DO

We have provided an example script to generate visualizations of hit patterns at fixed momentum (user provided), and over various theta in 5 degree bins. This will be done for both pions and kaons within the same script by default.

```
python plot_PDF.py --config config/config.json --method {} --momentum {} --model_type {} --fine_grained_prior --fs_support {}
```

| Argument               | Type     | Default        | Description                                                             |
|------------------------|----------|---------------|--------------------------------------------------------------------------|
| `--config`             | `str`    | `config.json` | Path to the config file                                                  |
| `--momentum`           | `float`  | `6`           | Momentum to generate (e.g., 6.5, 1.25)                                   |
| `--model_type`         | `str`    | `"NF"`        | Which model to use (`NF`,`CNF`,`FlowMatching`,`Score`,`DDPM`)            |
| `--fine_grained_prior` | `flag`   | `False`       | Enable fine-grained prior (just include the flag to activate the option) |
| `--fs_support`         | `float`  | `False`       | Number of photons to generate per PDF (default 200k)                     |


# Citation

```
@misc{giroux2025generativemodelsfastsimulation,
      title={Generative Models for Fast Simulation of Cherenkov Detectors at the Electron-Ion Collider}, 
      author={James Giroux and Michael Martinez and Cristiano Fanelli},
      year={2025},
      eprint={2504.19042},
      archivePrefix={arXiv},
      primaryClass={physics.ins-det},
      url={https://arxiv.org/abs/2504.19042}, 
}
```