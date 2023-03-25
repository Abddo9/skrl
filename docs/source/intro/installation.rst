Installation
============

In this section, you will find the steps to install the library, troubleshoot known issues, review changes between versions, and more.

.. raw:: html

    <br><hr>

Prerequisites
-------------

**skrl** requires Python 3.6 or higher and the following libraries (they will be installed automatically):

    * `gym <https://www.gymlibrary.dev>`_ / `gymnasium <https://gymnasium.farama.org/>`_
    * `tqdm <https://tqdm.github.io>`_
    * `packaging <https://packaging.pypa.io>`_
    * `torch <https://pytorch.org>`_ 1.8.0 or higher
    * `tensorboard <https://www.tensorflow.org/tensorboard>`_

.. raw:: html

    <br><hr>

Library Installation
--------------------

.. raw:: html

    <br>

Python Package Index (PyPI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install **skrl** with pip, execute:

    .. code-block:: bash

        pip install skrl

.. raw:: html

    <br>

GitHub repository
^^^^^^^^^^^^^^^^^

Clone or download the library from its GitHub repository (https://github.com/Toni-SM/skrl)

    .. code-block:: bash

        git clone https://github.com/Toni-SM/skrl.git
        cd skrl

* **Install in editable/development mode** (links the package to its original location allowing any modifications to be reflected directly in its Python environment)

    .. code-block:: bash

        pip install -e .

* **Install in the current Python site-packages directory** (modifications to the code downloaded from GitHub will not be reflected in your Python environment)

    .. code-block:: bash

        pip install .

.. raw:: html

    <br><hr>

Troubleshooting
---------------

To ask questions or discuss about the library visit skrl's GitHub discussions

.. centered:: https://github.com/Toni-SM/skrl/discussions

Bug detection and/or correction, feature requests and everything else are more than welcome. Come on, open a new issue!

.. centered:: https://github.com/Toni-SM/skrl/issues

.. raw:: html

    <br><hr>

Known issues
------------

1. When using the parallel trainer with PyTorch 1.12

    See PyTorch issue `#80831 <https://github.com/pytorch/pytorch/issues/80831>`_

    .. code-block:: text

        AttributeError: 'Adam' object has no attribute '_warned_capturable_if_run_uncaptured'

.. raw:: html

    <br><hr>

Changelog
---------

.. literalinclude:: ../../../CHANGELOG.md
    :language: markdown
