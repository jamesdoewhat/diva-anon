# DIVA anon

## Setup

These setup instructions were written for Ubuntu 20.04 with CUDA 12. 
Ensure that you also have CUDA toolkit installed, as well as the latest version of Anaconda. 
Additionally, we use virtual displays for certain environments, for which you will need to install Xvfb via:

```bash
sudo apt install xvfb
```

Clone repository:

```bash
git clone <anonrepo>
```

Upgrade pip if necessary (developed with version `22.3.1`):

```bash
pip install --upgrade pip
``` 

Install SWIG (required for `Box2D`), if not already installed (below is `Ubuntu`-specific command): 

```bash
sudo apt install swig
```

Run the following setup script to create a `conda` environment and to install the relevant dependencies:

```bash
. ./setup.sh
```

> __Note__: `mpi4py` is installed via `conda` before the rest of the requirements because
pip was having trouble installing the package at the time of writing.
`Tensorflow` is installed just before `mpi4py` because it is a required dependency.
`Pytorch` installation via `pip` was raising all sorts of dependency errors, so
we are using `conda` for now. The rest of the dependencies are then installed
via `pip` using the last line.

> __Note__: If you get the following error:
> ```bash
> AttributeError: module '_Box2D' has no attribute 'RAND_LIMIT_swigconstant'
> ```
> Then try: 
> ```bash
> pip uninstall box2d-py
> pip install box2d-py
> ```

## Reproducing main results

Each of the following commands use just one seed---you can change this value
to run over multiple seeds.

### GridNav

To reproduce results for the GridNav environment, run:

```bash
# DIVA results
python main.py diva_gridnav_main_results --env-type toygrid_qd_varibad --gt-type a24 --qd-initial-population 5000 --qd-warm-start-updates 200_000 --qd-warm-start-only True --qd-use-constant-mutations True --qd-mutations-constant 4 --qd-stepwise-mutations True --seed 123

# Robust PLR
python main.py rplr_gridnav_main_results --env-type toygrid_rplr_varibad --gt-type a24 --qd-updates-per-iter 4 --seed 123

# ACCEL results
python main.py accel_gridnav_main_results --env-type toygrid_accel_varibad --gt-type a24 --qd-updates-per-iter 4 --qd-use-constant-mutations True --qd-mutations-constant 4 --qd-stepwise-mutations True --seed 123

# DR results
python main.py dr_gridnav_main_results --env-type toygrid_gen_varibad --gt-type a24 --seed 123

# Oracle results
python main.py ods_gridnav_main_results --env-type toygrid_varibad --gt-type a24 --seed 123
```



### Alchemy

```bash
# DIVA results
python main.py diva_alchemy_main_results --env-type alchemy_qd_varibad --gt-type s8-d-d-d-c6 --qd-emitter-type me --qd-mutation-percentage 0.02 --qd-update-sample-mask True --qd-init-archive-dims 1 1 1 1 100 300 --qd-archive-dims 1 1 5 1 150 150 --qd-gt-diversity-objective False --qd-init-warm-start-updates 80_000 --qd-measures stone_reflection stone_rotation parity_first_stone parity_first_potion latent_state_diversity average_manhattan_to_optimal --qd-warm-start-updates 80_000 --qd-warm-start-only False --seed 123

# RPLR results
python main.py rplr_alchemy_main_results --env-type alchemy_rplr_varibad --gt-type s8-d-d-d-c6 --qd-updates-per-iter 2 --seed 123

# ACCEL results
python main.py accel_alchemy_main_results --env-type alchemy_accel_varibad --gt-type s8-d-d-d-c6 --qd-updates-per-iter 2 --seed 123

# DR results
python main.py dr_alchemy_main_results --env-type alchemy_gen_varibad --gt-type s8-d-d-d-c6 --seed 123

# Oracle results
python main.py ods_alchemy_main_results --env-type alchemy_varibad --gt-type s8-d-d-d-c6 --seed 123

```


### Racing


```bash
# DIVA results
python main.py diva_racing_main_results --env-type racing_qd_varibad --num-frames 2e7 --gt-type CP-32 --qd-update-sample-mask True --qd-es-sigma0 0.01 --qd-measures total_angle_changes com_x --qd-init-archive-dims 500 500 --qd-archive-dims 500 500 --qd-meas-alignment-objective True --qd-meas-alignment-measures com_y var_y --qd-randomize-objective True --qd-init-warm-start-updates 5e4 --qd-warm-start-updates 2e5 --qd-warm-start-only False --seed 123

# Robust PLR results
python main.py rplr_racing_main_results --env-type racing_rplr_varibad --num-frames 2e7 --gt-type CP-32 --qd-updates-per-iter 2 --seed 123

# ACCEL results
python main.py accel_racing_main_results --env-type racing_accel_varibad --num-frames 2e7 --gt-type CP-32 --qd-updates-per-iter 2 --seed 123

# DR results
python main.py dr_racing_main_results --env-type racing_gen_varibad --num-frames 2e7 --gt-type CP-32 --seed 123

# Oracle results
python main.py ods_racing_main_results --env-type racing_varibad --num-frames 2e7 --gt-type CP-32 --seed 123
```


For F1 results, set all else equal, but use `--test-env-name CarRacing-F1-v0`. For DIVA, you will first need
to run with `--qd-warm-start-only True`, and save the archive. Then you can run with the F1 argument, but using
`--qd-load-archive-from <saved_archive_name>`. This roundabout method is how we produced the results, and we haven't updated the code to makee this easier; for final
code release we will streamline this process.


For DIVA+ results, run:

```bash 
python main.py diva_racing_divaplus_results --env-type racing_qd_varibad --num-frames 2e7 --gt-type CP-32 --qd-plr-integration True --qd-load-archive-from <saved archive from previous run> --plr-env-generator gen --plr-replay-prob 1.0 --plr-level-replay-rho 0.0 --qd-warm-start-updates 2 --qd-load-archive-run-index 0 --qd-use-two-stage-ws False --qd-updates-per-iter 1 --qd-update-interval 4 --qd-log-interval 50 --qd-initial-population 0 --qd-batch-size 2 --qd-num-emitters 2 --qd-no-sim-objective False --qd-warm-start-no-sim-objective False --qd-use-plr-for-training True --seed 123
```

You will have needed to run a previous DIVA run with `--qd-warm-start-only True` to save the archive, and then
specify the archive loading location in the command above. This requires integration with W\&B currently,
but for the final code release we will streamline this process and make it more accessible.

# Licenses

This codebase is primarly based off of VariBAD, which uses an MIT License, which we are in compliance with.
