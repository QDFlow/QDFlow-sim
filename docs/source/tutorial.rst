QDFlow Tutorial
===============

QDFlow is an open-source physics simulator for Quantum Dot arrays that generates
realistic Charge-Stability Diagrams (CSDs) which mimic experimental data, along
with ground-truth labels of the charge state at each point in the CSD.

This tutorial will demonstrate how to use QDFlow to generate realistic, diverse,
synthetic datasets suitable for training and benchmarking Machine Learning
models as well as other applications in quantum dot research.

This tutorial requires the `tutorial_helper.py` file, which
contains helper functions used to streamline plotting the results. This file is
available from the `QDFlow GitHub repository <https://github.com/QDFlow/QDFlow-sim>`_,
along with the tutorial notebook itself.

.. toctree::
   :maxdepth: 1

   _tutorial/getting_started
   _tutorial/generating_datasets
   _tutorial/distributions
   _tutorial/custom_distributions
   _tutorial/correlated_distributions
   _tutorial/adding_noise
   _tutorial/ray_data
   _tutorial/physics_simulation
   _tutorial/thomas_fermi_solver
   _tutorial/capacitance_model
