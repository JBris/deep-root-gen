DeepRootGen
===========

This page describes how to install DeepRootGen, run a root system simulation,
work with its exported graph data, and contribute to the project. For the
scientific background and an overview of the modelling framework, see the
Introduction page.

Installation
------------

Requirements
~~~~~~~~~~~~

DeepRootGen targets Python ``>=3.10,<3.11`` and uses Poetry to manage its
project dependencies. For reproducible installations, use the committed
``poetry.lock`` file.

Install with Poetry
~~~~~~~~~~~~~~~~~~~

Clone the repository:

.. code-block:: bash

   git clone https://github.com/Plant-Food-Research-Open/deep-root-gen.git
   cd deep-root-gen

Install Poetry if necessary and then install the project:

.. code-block:: bash

   python -m pip install poetry
   poetry install

Run commands inside the Poetry environment:

.. code-block:: bash

   poetry run python

For more detailed setup instructions, refer to the
`development and setup guide <https://github.com/Plant-Food-Research-Open/deep-root-gen/blob/main/docs/DEVELOPMENT.md>`_.

Quickstart
----------

The following example creates a stochastic three-dimensional root system and
exports its hierarchical graph as separate node and edge tables.

.. code-block:: python

   from deeprootgen.data_model import RootSimulationModel
   from deeprootgen.model import RootSystemSimulation

   parameters = RootSimulationModel(
       random_seed=42,
       max_order=3,
       root_ratio=0.5,
       fine_root_threshold=0.15,
       outer_root_num=4,
       inner_root_num=2,
       min_primary_length=15.0,
       max_primary_length=40.0,
       base_diameter=0.5,
       diameter_reduction=0.2,
       apex_diameter=0.05,
       min_sec_root_num=1,
       max_sec_root_num=3,
       growth_sec_root=0.1,
       min_sec_root_length=3.0,
       max_sec_root_length=15.0,
       segments_per_root=6,
       length_reduction=0.7,
       root_vary=10.0,
       interbranch_distance=2.0,
       mechanical_constraints=0.0,
       root_tissue_density=0.15,
       gravitropism=0.1,
       t=1,
       enable_soil=False,
       soil_layer_height=10.0,
       soil_n_layers=10,
       soil_layer_width=10.0,
       soil_n_cols=10,
   )

   simulation = RootSystemSimulation(random_seed=42)
   simulation.run(parameters)

   node_df, edge_df = simulation.G.as_df()

   node_df.to_csv("root_nodes.csv", index=False)
   edge_df.to_csv("root_edges.csv", index=False)

   print(f"Nodes: {len(node_df)}; edges: {len(edge_df)}")

A fixed random seed makes it possible to reproduce an individual stochastic
realisation, subject to the software environment and model implementation.
Different seeds can produce different root architectures under the same
parameter configuration.

Working with Root Graph Data
----------------------------

The ``RootSystemGraph.as_df()`` method exports a node table and an edge table.
The node schema is defined by ``RootNodeModel`` in
``deeprootgen/data_model/simulation_data_models.py``. The tables can be used
for plotting, architectural statistics, or external analysis.

Node table
~~~~~~~~~~

.. list-table:: Node fields
   :header-rows: 1
   :widths: 28 72

   * - Column
     - Meaning
   * - ``node_id``
     - Unique graph node identifier.
   * - ``parent_id``
     - Identifier of the parent node; the base node uses ``-1``.
   * - ``plant_id``
     - Plant identifier.
   * - ``organ_id``
     - Root organ identifier.
   * - ``order``
     - Root branching order; base node is order zero.
   * - ``segment_rank``
     - Segment position within a root organ.
   * - ``x``, ``y``, ``z``
     - Spatial coordinates (cm).
   * - ``diameter``, ``length``
     - Recorded segment dimensions (cm).
   * - ``root_tissue_density``
     - Root tissue density (g/cm³).
   * - ``root_type``
     - Classification such as ``structural``, ``fine`` or ``base``.
   * - ``order_type``, ``position_type``
     - Additional classifications.
   * - ``simulation_tag``
     - Simulation grouping tag.
   * - ``invalid_root``
     - Flag identifying segments retained in the export but marked invalid.

Edge table
~~~~~~~~~~

.. list-table:: Edge fields
   :header-rows: 1
   :widths: 28 72

   * - Column
     - Meaning
   * - ``parent_id``
     - Parent node identifier.
   * - ``child_id``
     - Child node identifier.

The node and edge tables jointly preserve the explicit parent-child
connectivity of the simulated root system. Reconstruct segment geometry from
the coordinates of each child and its parent rather than assuming the rows
are already ordered into root paths.

.. note::

   The export can retain segments marked ``invalid_root=True``. Apply a
   consistent validity filter when calculating summary statistics. Some
   statistics use recorded segment lengths; geometric analyses may instead
   use the distance between parent and child coordinates. Whole-root totals
   may also differ from lengths clipped to a finite soil voxel grid.

The current export should not be assumed to use older field conventions such
as ``id``, ``parent``, a combined ``coordinates`` field, or numeric root types.
Check downstream data consumers before translating or renaming identifiers.

Development
-----------

Repository resources
~~~~~~~~~~~~~~~~~~~~

Useful starting points for contributors include:

* `Development and setup guide <https://github.com/Plant-Food-Research-Open/deep-root-gen/blob/main/docs/DEVELOPMENT.md>`_
* `Architecture and code map <https://github.com/Plant-Food-Research-Open/deep-root-gen/blob/main/docs/ARCHITECTURE.md>`_
* `Review notes <https://github.com/Plant-Food-Research-Open/deep-root-gen/blob/main/docs/REVIEW_NOTES.md>`_
* `Agent instructions <https://github.com/Plant-Food-Research-Open/deep-root-gen/blob/main/AGENTS.md>`_

Testing
~~~~~~~

Run the project's existing test suite with:

.. code-block:: bash

   poetry run pytest

Passing the existing automated tests does not, by itself, establish the
scientific validity of generated root systems. Changes to the simulator or
statistics should also be checked for geometric consistency, graph integrity,
root branching behaviour, and agreement between summary calculations.

Pre-commit hooks
~~~~~~~~~~~~~~~~

Install and run the configured hooks using:

.. code-block:: bash

   poetry run pre-commit install
   poetry run pre-commit run --all-files

Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The online documentation is available from the
`DeepRootGen website <https://Plant-Food-Research-Open.github.io/deep-root-gen/>`_.
The documentation uses Sphinx and the Read the Docs theme.

From the repository root, build the HTML pages using:

.. code-block:: bash

   cd docs
   make html

Open ``docs/build/html/index.html`` to view the generated documentation.

Community and Support
---------------------

For questions, bug reports, and feature requests, use the repository's
`GitHub Issues <https://github.com/Plant-Food-Research-Open/deep-root-gen/issues>`_.
For proposed code changes, see
`GitHub Pull Requests <https://github.com/Plant-Food-Research-Open/deep-root-gen/pulls>`_.

When reporting a simulation issue, include the input parameters, random seed,
Python environment, relevant errors, and steps to reproduce the behaviour.

Contributions are welcome. Review the project's
`contribution guidelines <https://github.com/Plant-Food-Research-Open/deep-root-gen/blob/main/docs/source/community/CONTRIBUTING.md>`_
and
`code of conduct <https://github.com/Plant-Food-Research-Open/deep-root-gen/blob/main/docs/source/community/CODE_OF_CONDUCT.md>`_
before contributing.

License
-------

DeepRootGen is distributed under the GNU General Public License v3.0 (GPL-3.0).
See the `LICENSE file <https://github.com/Plant-Food-Research-Open/deep-root-gen/blob/main/LICENSE>`_
for the full licence terms.
