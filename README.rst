.. image:: https://raw.githubusercontent.com/felavila/tuskitoo/main/docs/source/_static/tuskitoo.png
   :alt: tuskitoo Logo
   :align: left
   :width: 400

Tuskitoo: Two-dim Spectra Kit Tool
==================================

Tuskitoo (Two-dim Spectra Kit Tool) is a Python 3 package that integrates a range of tools for advanced two-dimensional spectral analysis. Originally developed for X-shooter data, it is flexible enough to be applied to spectra from other instruments as well. The package streamlines the spectral extraction process and implements PCA-based sky subtraction, providing a comprehensive framework for analyzing complex spectroscopic observations.

Features
========

- **Spectral Extraction:** Tools for accurate extraction of spectra from raw two-dimensional observations.
- **PCA Sky Subtraction:** Uses Principal Component Analysis (PCA) to effectively subtract sky background.
- **Telluric Correction:** Applies telluric correction to 1D or 2D spectra.
- **Visualization:** Provides straightforward visualization of both 1D and 2D spectra.

Installation
============

Install Tuskitoo locally using the following command:

.. code-block:: shell

   pip install -e .

Prerequisites
=============

You need to have Python == 3.10 installed.

References
==========

Tuskitoo is based on methodologies presented in:

**Melo, A. et al. (2021). First black hole mass estimation for the quadruple lensed system WGD2038-4008. Astronomy & Astrophysics, 656, A108.**
Available at: `ADS Abstract <https://ui.adsabs.harvard.edu/abs/2021A%26A...656A.108M/abstract>`_

License
=======

* Free software: GNU Affero General Public License v3.0
