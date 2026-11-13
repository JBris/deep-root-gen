
Introduction
============

*A stochastic simulation framework for generating, analysing and
digitally reconstructing three-dimensional root system architectures,
with support for model calibration and simulation-based inference.*

Overview
--------

Root system architecture (RSA) plays an important role in plant growth
and development, influencing soil exploration, water and nutrient
uptake, and interactions with the surrounding environment. However,
characterising root systems under field conditions remains challenging
due to their spatial complexity, below-ground location, and the cost
and destructive nature of many root measurement techniques.

DeepRootGen provides a computational framework for generating
synthetic three-dimensional root system architectures using a
stochastic, parameterised root simulation model. The model represents
individual root organs and segments within a hierarchical graph,
allowing their geometry, topology and spatial distribution to be
analysed.

The framework is intended to support the digital reconstruction of
root systems using observational data collected from the field. By
combining mechanistic assumptions about root development with
statistical calibration and inference methods, DeepRootGen provides
a means of investigating plausible root architectures consistent
with available measurements.

DeepRootGen has been developed with an emphasis on woody perennial
root systems, including horticultural applications involving apple
and grapevine rootstocks. The underlying simulator is parameterised
to support different root system configurations and can be
recalibrated for specific applications.


Features and Functionality
--------------------------

DeepRootGen provides the following functionality:

* **Stochastic 3D root simulation:** Generate synthetic root system
  architectures with configurable root numbers, lengths, diameters,
  branching behaviour and growth orientations.

* **Hierarchical graph representation:** Represent root organs and
  segments using explicit parent-child relationships, branching
  orders and segment identifiers.

* **Root branching and development:** Simulate primary and
  higher-order roots, including stochastic lateral-root attachment
  locations and configurable interbranch distances.

* **Architectural analysis:** Calculate summary statistics describing
  root length, diameter, branching patterns, orientation, spatial
  extent and root length density.

* **Spatial aggregation:** Aggregate simulated root segments onto
  three-dimensional soil voxel grids for comparison against field
  observations.

* **Model calibration:** Estimate root system parameters using
  optimisation and statistical discrepancy measures.

* **Sensitivity analysis:** Investigate the influence of model
  parameters on simulated architectural characteristics.

* **Stochastic ensemble simulation:** Generate multiple plausible
  root architectures under a common parameter configuration to
  investigate variability between realisations.

* **Visualisation:** Explore root architectures and summary
  statistics through interactive three-dimensional visualisations
  and statistical plots.

* **Experiment management:** Support reproducible simulation
  workflows, experiment tracking and model artefact management.

The repository also contains infrastructure for simulation-based
inference and generative modelling. The availability and configuration
of individual inference workflows depend on the installed
dependencies and repository revision.


Model Overview
-------------

DeepRootGen uses a stochastic root growth simulator to construct
root system architectures from a configurable set of parameters.


Root Architecture Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A simulated root system consists of primary roots originating from
the plant base, with lateral roots emerging from their parent organs
according to the specified branching configuration.

The model supports several aspects of root development:

* Number and arrangement of primary roots.
* Root organ lengths and diameters.
* Maximum branching order.
* Number and length of lateral roots.
* Stochastic variation in root growth.
* Interbranch distances and lateral-root attachment locations.
* Root growth orientation and gravitropism.
* Classification of structural and fine roots.

Individual simulation runs may produce different architectures even
when the input parameters are identical. Random seeds can be
specified to reproduce individual stochastic realisations.


Hierarchical Root Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each root system is represented as a graph containing root nodes
and their relationships.

Root organs are associated with unique identifiers, while individual
segments retain their position within each organ. The hierarchy
distinguishes primary roots from higher-order lateral roots and
preserves the attachment relationships between parent and daughter
organs.

This representation enables analysis of both root geometry and
topology, including branching orders, root lengths, branching
frequencies and spatial orientations.

The root graph can be exported as separate node and edge tables,
allowing integration with downstream analytical and visualisation
workflows.


Root System Architecture Analysis
---------------------------------

DeepRootGen provides summary statistics for characterising simulated
root systems and comparing them with observational measurements.

Root Morphology
~~~~~~~~~~~~~~~

Root morphology can be described using statistics such as:

* Total root length and root number.
* Root length and diameter distributions.
* Total and average root volume.
* Root dry mass and root tissue density.
* Specific root length.
* Root dry mass by diameter class.

Statistics may be calculated for individual root organs, branching
orders or the complete root system, depending on the selected method.


Root Branching and Orientation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The hierarchical root graph provides information for analysing root
topology and spatial growth patterns.

Relevant characteristics include:

* Root number and length by branching order.
* Branching frequency and interbranch distance.
* Parent-daughter branching angles.
* Root elongation directions.
* Segment inclination and azimuth.
* Root-level growth orientation.

These metrics can be used to investigate how model parameters
influence the geometry and structural organisation of simulated
root systems.


Spatial Root Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~

DeepRootGen supports spatial analysis of root systems, including:

* Maximum rooting depth.
* Horizontal root spread and radial extent.
* Root length distributions by soil depth.
* Horizontal and radial root length distributions.
* Convex hull area and volume.
* Three-dimensional root length density.

Spatial distributions are particularly useful when comparing
simulated architectures with soil-core, trench, excavation or
voxel-based observations.


Root Length Density
~~~~~~~~~~~~~~~~~~~

Root length density (RLD) describes the amount of root length
contained within a specified soil volume:

.. math::

   \mathrm{RLD} =
   \frac{\mathrm{Root\ length\ (cm)}}
        {\mathrm{Soil\ volume\ (cm^3)}}

DeepRootGen supports the calculation of simulated RLD using a
three-dimensional soil voxel grid.

Root segments are intersected with voxel boundaries so that their
lengths can be allocated to the corresponding soil volumes. Both
recorded and geometrically derived segment lengths are supported.

The resulting voxel-level data can be aggregated by soil layer,
row or column, enabling comparison with observed RLD distributions.

When comparing simulated and observed data, the coordinate system,
voxel boundaries, root validity filters and length calculation
method should be consistent.


Model Calibration and Sensitivity Analysis
------------------------------------------

DeepRootGen includes workflows for estimating simulation parameters
and evaluating model behaviour.


Model Calibration
~~~~~~~~~~~~~~~~~

Model calibration aims to identify parameter configurations that
produce simulated root characteristics consistent with observational
measurements.

The repository includes an optimisation workflow using
`Optuna <https://optuna.org/>`_, with support for specifying parameter
bounds and minimising statistical discrepancies between simulated
and observed summary statistics.

Calibration targets may include root length, diameter, rooting depth,
branching characteristics or spatial root length density.

Because the simulator is stochastic, repeated simulations can be
used to estimate the expected discrepancy for a candidate parameter
configuration.

Independent observational units and held-out datasets can be used
to evaluate predictive performance outside the calibration sample.


Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~

Sensitivity analysis can be used to investigate how uncertainty or
variation in simulation parameters affects root architecture
characteristics.

The repository includes a sensitivity analysis workflow based on
`SALib <https://salib.readthedocs.io/>`_, supporting Sobol
sensitivity indices.

First-order and total-order Sobol indices provide information about
the influence of individual parameters and their interactions on
a selected model output.

Sensitivity analysis may be applied to statistics such as:

* Total root length and root number.
* Maximum rooting depth.
* Root branching frequency.
* Mean root diameter.
* Spatially aggregated root length density.
* Replicate-to-replicate variability in root system characteristics.

For stochastic analyses, repeated simulations are needed to
distinguish parameter-driven changes in model output from variation
associated with random simulation realisations.


Stochastic Root Architecture Ensembles
--------------------------------------

Repeated simulations can be used to generate an ensemble of plausible
root architectures under a common parameter configuration.

For a calibrated parameter vector, each stochastic realisation
represents one possible root system generated by the model.

Ensemble analysis can be used to estimate:

* Expected root system characteristics.
* Variation in total root length and root number.
* Variation in root branching patterns and rooting depth.
* Root length and diameter distributions.
* Voxel-wise mean root length density.
* Voxel-wise standard deviation and occupancy probability.

Each ensemble member retains its own hierarchical root graph.
Consequently, the ensemble may be analysed as a collection of
individual architectures or aggregated into spatial probability
distributions.

An ensemble-mean RLD field does not necessarily correspond to a
single physically realisable root architecture. A representative
simulation may instead be selected based on its proximity to the
ensemble-mean spatial distribution.

These ensembles describe stochastic variability conditional on the
specified model parameters. Additional analysis is required to
account for uncertainty in the estimated parameters themselves.


Contacts
--------

* **DeepRootGen Developer:** James Bristow
* **DeepRootGen Project Supervisor:** Junqi Zhu

Funding and Acknowledgements
----------------------------

This work was funded by the New Zealand Ministry of Business,
Innovation, and Employment (MBIE) through the Strategic Science
Investment Fund (contract number C11X1702) as part of the Digital
Horticultural Systems Ngā Pou Rangahau, a Growing Futures™ programme
at Plant & Food Research.

We sincerely thank the Fruit Physiology Group, orchard managers,
and all those who contributed their time and effort to the
management and operation of the orchards. Their contributions
were essential to the collection of observational data and the
development of the modelling framework.

We also acknowledge the wider Digital Horticultural Systems team
for their support and contributions to the research and development
of DeepRootGen.
