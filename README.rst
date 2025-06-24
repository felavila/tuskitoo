.. raw:: html

    <div style="text-align: center">

.. image:: docs/images/tuskitoo.png
   :alt: Tuskitoo Logo
   :width: 400

.. raw:: html

    </div>

=======
Tuskitoo: Two-dim Spectra Kit Tool
=======
.. .https://python-package-generator.readthedocs.io/en/master/
.. .| |build| |coverage| |docs| |maintainability| |codacy| |tech-debt|
.. .| |release_version| |wheel| |supported_versions| |commits_since_specific_tag_on_master| |commits_since_latest_github_release|
.. .| |pypi_stats| |ossf| |ruff| |black| |gh-lic||

Tuskitoo is a Python 3 package that integrates a range of tools for advanced two-dimensional 
spectral analysis. It streamlines the process of spectral extraction, employs PCA for robust 
sky subtraction, providing researchers with a comprehensive framework to analyze complex spectral data.

Features
=======

- **Spectral Extraction:** Tools designed to accurately extract spectral data from raw two-dimensional observations.
- **PCA Sky Subtraction:** Implements Principal Component Analysis techniques to effectively subtract sky background noise, enhancing data quality.



Installation
=======
.. . old Tudspek name 
Install  Tuskitoo locally using the following command:

.. code-block:: shell
    
    pip install -e .

Prerequisites
=============

You need to have `Python` == 3.10 installed .

References
=======

Tuskitoo is based on methodologies presented in the following paper(s):

**Melo, A. et al. (2021). First black hole mass estimation for the quadruple lensed system WGD2038-4008.
Astronomy & Astrophysics, Volume 656, id.A108, 10 pp.**  
Available at: `ADS Abstract <https://ui.adsabs.harvard.edu/abs/2021A%26A...656A.108M/abstract>`_  


License
=======

|gh-lic|

* `GNU Affero General Public License v3.0`_


License
=======

* Free software: GNU Affero General Public License v3.0


.. BADGE ALIASES

.. Build Status
.. Github Actions: Test Workflow Status for specific branch <branch>

.. |build| image:: https://img.shields.io/github/actions/workflow/status/boromir674/cookiecutter-python-package/test.yaml?link=https%3A%2F%2Fgithub.com%2Fboromir674%2Fcookiecutter-python-package%2Factions%2Fworkflows%2Ftest.yaml%3Fquery%3Dbranch%253Amaster
   :alt: GitHub Workflow Status (with event)

.. build target https://github.com/boromir674/cookiecutter-python-package/actions/workflows/test.yaml?query=branch%3Amaster


.. Documentation

.. |docs| image:: https://img.shields.io/readthedocs/python-package-generator/master?logo=readthedocs&logoColor=lightblue
    :alt: Read the Docs (version)
    :target: https://python-package-generator.readthedocs.io/en/master/

.. Code Coverage

.. |coverage| image:: https://img.shields.io/codecov/c/github/boromir674/cookiecutter-python-package/master?logo=codecov
    :alt: Codecov
    :target: https://app.codecov.io/gh/boromir674/cookiecutter-python-package

.. PyPI

.. |release_version| image:: https://img.shields.io/pypi/v/cookiecutter_python
    :alt: Production Version
    :target: https://pypi.org/project/cookiecutter-python/

.. |wheel| image:: https://img.shields.io/pypi/wheel/cookiecutter-python?color=green&label=wheel
    :alt: PyPI - Wheel
    :target: https://pypi.org/project/cookiecutter-python

.. |supported_versions| image:: https://img.shields.io/pypi/pyversions/cookiecutter-python?color=blue&label=python&logo=python&logoColor=%23ccccff
    :alt: Supported Python versions
    :target: https://pypi.org/project/cookiecutter-python

.. |pypi_stats| image:: https://img.shields.io/pypi/dm/cookiecutter-python?logo=pypi&logoColor=%23849ED9&color=%23849ED9&link=https%3A%2F%2Fpypi.org%2Fproject%2Fcookiecutter-python%2F&link=https%3A%2F%2Fpypistats.org%2Fpackages%2Fcookiecutter-python
    :alt: PyPI - Downloads
    :target: https://pypistats.org/packages/cookiecutter-python

.. Github Releases & Tags

.. |commits_since_specific_tag_on_master| image:: https://img.shields.io/github/commits-since/boromir674/cookiecutter-python-package/v2.5.0/master?color=blue&logo=github
    :alt: GitHub commits since tagged version (branch)
    :target: https://github.com/boromir674/cookiecutter-python-package/compare/v2.5.0..master

.. |commits_since_latest_github_release| image:: https://img.shields.io/github/commits-since/boromir674/cookiecutter-python-package/latest?color=blue&logo=semver&sort=semver
    :alt: GitHub commits since latest release (by SemVer)


.. LICENSE (eg AGPL, MIT)
.. Github License

.. |gh-lic| image:: https://img.shields.io/github/license/boromir674/cookiecutter-python-package
    :alt: GitHub
    :target: https://github.com/boromir674/cookiecutter-python-package/blob/master/LICENSE


.. Free/Libre Open Source Software
.. Open Source Software Best Practices

.. |ossf| image:: https://bestpractices.coreinfrastructure.org/projects/5988/badge
    :alt: OpenSSF
    :target: https://bestpractices.coreinfrastructure.org/en/projects/5988


.. CODE QUALITY

.. Codacy
.. Code Quality, Style, Security

.. |codacy| image:: https://app.codacy.com/project/badge/Grade/5be4a55ff1d34b98b491dc05e030f2d7
    :alt: Codacy
    :target: https://app.codacy.com/gh/boromir674/cookiecutter-python-package/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=boromir674/cookiecutter-python-package&amp;utm_campaign=Badge_Grade


.. Code Climate CI
.. Code maintainability & Technical Debt

.. |maintainability| image:: https://api.codeclimate.com/v1/badges/1d347d7dfaa134fd944e/maintainability
   :alt: Maintainability
   :target: https://codeclimate.com/github/boromir674/cookiecutter-python-package/

.. |tech-debt| image:: https://img.shields.io/codeclimate/tech-debt/boromir674/cookiecutter-python-package
    :alt: Code Climate technical debt
    :target: https://codeclimate.com/github/boromir674/cookiecutter-python-package/

.. Ruff linter for Fast Python Linting

.. |ruff| image:: https://img.shields.io/badge/code%20style-ruff-000000.svg
    :alt: Ruff
    :target: https://docs.astral.sh/ruff/

.. Code Style with Black

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :alt: Black
    :target: https://github.com/psf/black
