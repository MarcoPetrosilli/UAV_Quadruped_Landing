<h1 align="center"> UAV Dynamic Landing Control</h1>

<p align="center">
  <img src="https://img.shields.io/badge/conda-env-green.svg" alt="Conda">
  <img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/control-MPC%20%2B%20CBF-success.svg" alt="Control">
  <img src="https://img.shields.io/badge/simulation-PyBullet-orange.svg" alt="Simulation">
</p>

> **Overview:** This document provides instructions on how to set up the Conda virtual environment required to run the control and the 3D physics simulations.

---

## 1. Creating the Conda Environment

Open your terminal, navigate to the root folder of the project (`UAV_Quadruped_Landing`), and create a new Conda environment named `drones` using Python 3.10:

```bash
conda create --name drones python=3.10 -y
```

## 2. Activate the Environment

```bash
conda activate drones
```

## 3. Installing Dependencies

```bash
- 1. Install core mathematical and optimization packages
conda install -c conda-forge numpy scipy cvxpy osqp matplotlib -y

- 2. Install 3D simulation frameworks via pip
pip install gym-pybullet-drones pybullet gymnasium
```

## Packages Details

| Package | Purpose in the Project |
| :--- | :--- |
| **`numpy`** & **`scipy`** | Core matrix operations, spatial calculations, and loading pre-computed reachable polytopes (`.npz`). |
| **`cvxpy`** & **`osqp`** | Formulating and solving the quadratic optimization problem (MPC) and spatial constraints (CBFs). |
| **`matplotlib`** | Plotting trajectories and visually debugging the controller behavior. |
| **`gym-pybullet-drones`**, **`pybullet`** & **`gymnasium`** | The 3D physics engine and simulation framework required for testing the drone landing algorithms. |

## 4. Deactivation

```bash
conda deactivate
```

## 5. Launching the Simulation

```bash
cd gym_pybullet_drones/examples/thesis/

python MPC_PID_hybrid_dynamic.py 
```
This is the current version of the simulation, which relies on the control law implementation that can be found at:

```bash
gym_pybullet_drones/control/MPCPIDHYControlDynamic.py.
```
  

The other files that can be found under the folder example/thesis are old versions of the control, such as:

| File | Description |
| :--- | :--- |
| **`control_to_wp.py//`** | Pure PID control |
| **`LOS_to_wp.py`** | PID control implementing LOS (Line of Sight) approach |
| **`MPC_Position.py`** | MPC for the entire mission |








