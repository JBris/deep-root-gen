Changelog
=========

1.0.0 - 2026/11/13
------------------

Added
^^^^^

* Added root number and root length distribution statistics, including summaries by branching order.
* Added root diameter distributions, including length-weighted diameter classes.
* Added root branching frequency, branching order, and branching angle statistics.
* Added root elongation direction and orientation statistics, including inclination and azimuth.
* Added rooting depth and horizontal spread metrics.
* Added vertical and horizontal root length distributions.
* Added functionality for estimating radial root length distributions.
* Added support for stochastic lateral-root attachment using interbranch distances.
* Added attachment nodes along parent roots while preserving hierarchical graph connectivity.

Improved
^^^^^^^^

* Improved three-dimensional root length density (RLD) calculations using geometric segment lengths and voxel intersections.
* Improved handling of root segments crossing soil voxel boundaries.
* Improved root length, diameter, volume, and mass calculations.
* Improved consistency between architectural statistics and voxel-based RLD aggregation.
* Improved handling of invalid root segments across summary statistics.
* Expanded the summary statistics registry to expose additional architectural metrics.

Fixed
^^^^^

* Corrected average root diameter calculations to use length-weighted means.
* Corrected depth distributions to represent root length rather than node counts.
* Corrected radial distributions to use horizontal Euclidean distance from the root origin.
* Corrected specific root length calculations and associated units.
* Improved handling of empty and degenerate root systems in statistical calculations.
* Corrected inconsistencies arising from recorded versus geometric segment lengths.

Changed
^^^^^^^

* Standardised architectural statistics around parent-child segment geometry.
* Updated selected summary statistics to require hierarchical root-node data.
* Updated depth and radial distribution comparison outputs to represent cumulative root-length fractions.
* Changed the default segment-length calculation for selected statistics to use geometric lengths.

Compatibility
^^^^^^^^^^^^^

* Retained existing summary statistic class names and primary calculation interfaces.
* Retained existing root length density aggregation methods.
* Some calculation defaults, input requirements, and comparison outputs have changed and may affect existing calibration workflows.

0.1.0 - 2024/08/17
------------------

Added
^^^^^

* Initial project structure and implementation.
* Initial stochastic root system architecture simulation functionality.
* Initial root graph representation and data models.
* Initial root system summary statistics and visualisation utilities.
* Initial calibration and sensitivity analysis infrastructure.