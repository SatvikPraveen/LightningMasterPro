.. docs/source/index.rst

LightningMasterPro Documentation
=================================

Welcome to LightningMasterPro, a comprehensive PyTorch Lightning learning and experimentation framework.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   modules
   datamodules
   callbacks
   notebooks
   api

Overview
--------

LightningMasterPro provides:

- **Modular Architecture**: Clean separation of data, models, and training logic
- **Multiple Domains**: Vision, NLP, Tabular, and Time Series examples
- **Educational Notebooks**: Progressive learning path from basics to advanced
- **Best Practices**: Production-ready patterns and configurations
- **Extensible Design**: Easy to add new modules and experiments

Quick Start
-----------

.. code-block:: bash

   pip install -e .
   python scripts/train.py --config configs/vision/classifier.yaml

Key Features
------------

- **Lightning Modules**: Domain-specific model implementations
- **Data Modules**: Synthetic data generation for experimentation  
- **Custom Callbacks**: EMA, SWA, and advanced checkpointing
- **Training Loops**: K-fold validation and curriculum learning
- **Configuration System**: YAML-based experiment management

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`