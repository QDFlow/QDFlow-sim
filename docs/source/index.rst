QDFlow documentation
====================

A Thomas-Fermi simulation for modeling quantum dot devices.
Generates charge stability diagrams complete with sensor readout, state labels, charge configuration, and noise.

For more information, see `arXiv:2509.13298 <https://arxiv.org/abs/2509.13298>`_.

Installation
============

Install the latest version of QDFlow:

.. code-block:: sh

   pip install qdflow

API
===

.. autosummary::
   :toctree: _autosummary
   :recursive:

   qdflow.generate
   qdflow.physics.simulation
   qdflow.physics.noise
   qdflow.util.distribution
