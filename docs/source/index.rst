LightningMasterPro Documentation
==================================

**LightningMasterPro** is a hands-on PyTorch Lightning learning framework. It pairs
20 educational notebooks with a small, fully tested library (``lmpro``) that shows
the idiomatic Lightning 2.x way to build modules, datamodules, callbacks, custom
training drivers and a ``LightningCLI``.

Quick start
-----------

.. code-block:: bash

   pip install -e ".[dev,export]"
   python scripts/train.py fit --config configs/vision/classifier.yaml
   lmpro fit --config configs/nlp/sentiment.yaml --trainer.max_epochs 3

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
