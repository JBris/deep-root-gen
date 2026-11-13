.. _grapevine-calibration-faq:

FAQ
===

This page responds to the five questions in the ``Questions`` worksheet of
``Paper2_data_preparation_draft_oct_checked.xlsx``. It refers to the workbook's
``README``, ``Data description``, ``what can my datasets inform``,
``possible model parameters``, ``RLD (x, y, z)``, ``Root size summary example``,
and ``Root angle summary example`` worksheets.

The intended application is reconstruction of a **plausible representative
root system for each site**, not an exact reconstruction of the excavated
subject vine. The source code interpretation below is based on the
``RootOrgan`` and ``RootSystemSimulation`` implementation supplied for review.
Where functionality has been proposed but not integrated, this is stated
explicitly. The public project is at
`DeepRootGen on GitHub <https://github.com/Plant-Food-Research-Open/deep-root-gen>`_.

1. What format should spatial RLD observations use? Is the feature available?
----------------------------------------------------------------------------

**Recommended exchange format:** a UTF-8 CSV file with **one row per sampled
core depth increment**, retaining the sampling location, depth, observed RLD,
sampling status, and soil volume. This is a proposed convention rather than
an existing mandatory DeepRootGen input schema.

The workbook already contains an appropriate source layout in the
``RLD (x, y, z)`` worksheet:

.. csv-table:: Example taken from the workbook, Site 1, ``tube4``
   :header: "site", "core_id", "x_cm", "y_cm", "z_mid_cm", "core_volume_cm3", "grapevine_rld_cm_cm^-3", "sampled"
   :widths: 12, 12, 9, 9, 12, 16, 23, 9

   "Site 1", "tube4", 8.5, -27, 5, 74.22, 0.487, "Y"
   "Site 1", "tube4", 8.5, -27, 15, 74.22, 0.503, "Y"
   "Site 1", "tube4", 8.5, -27, 25, 74.22, 0.255, "Y"
   "Site 1", "tube4", 8.5, -27, 145, 74.22, 0, "N"

Retain ``site`` and ``core_id`` as identifiers. Their names do not affect
the numerical model, but are necessary for matching repeated increments and
site-specific fitting. ``vine_dw_g`` may be retained as an optional extra
column. A descriptive filename such as
``grapevine_core_rld_observations.csv`` is suggested; the code has no
special required filename.

**Coordinate convention:** the workbook uses positive ``z_mid_cm`` for depth
below the surface, whereas the simulator uses ``z=0`` at ground level and
negative ``z`` below ground. Each increment is 10 cm thick. A workbook
midpoint of 5 cm therefore corresponds to ``z_lower_bound=-10`` and
``z_upper_bound=0`` in the simulator; 15 cm corresponds to -20 to -10 cm.
The ``x`` and ``y`` coordinates are measured relative to the subject vine
trunk. Establish their orientation relative to the vine row and the model
axes before fitting.

.. code-block:: python

   import pandas as pd

   observations = pd.read_csv("grapevine_core_rld_observations.csv")
   observations = observations.loc[
       observations["sampled"].str.upper().eq("Y")
   ].copy()

   # Workbook depths are positive downward; model z is negative downward.
   observations["z_lower_bound"] = -(observations["z_mid_cm"] + 5.0)
   observations["z_upper_bound"] = -(observations["z_mid_cm"] - 5.0)
   observations["observed_rld_cm_cm3"] = (
       observations["grapevine_rld_cm_cm^-3"]
   )

A **measured zero with ``sampled='Y'`` remains a valid observation**.
Rows marked ``sampled='N'`` must be excluded rather than interpreted as
zero RLD. This matters for the 145 and 155 cm increments in the workbook
example, and for unsampled shallow increments at Site 3.

**Implementation status:** the existing spatial RLD tools can calculate
simulated voxel-level values. The model's older calibration input
(``SummaryStatisticsModel``) accepts scalar observations and optional depth
bounds, but not a complete field-core location. A proposed extension adds
``x``, ``y``, depth bounds and ``core_radius`` to each observation, together
with spatial matching in the calibration objective. The proposal has **not
been verified as merged** into the application; the CSV above is not yet a
plug-and-play calibration input.

2. Can the code and parameter definitions be reviewed before fieldwork?
-----------------------------------------------------------------------

**Yes.** The source is in the public
`DeepRootGen repository <https://github.com/Plant-Food-Research-Open/deep-root-gen>`_.
The relevant code to review is ``RootOrgan`` and ``RootSystemSimulation``
(in ``deeprootgen/model``), the ``RootSimulationModel`` schema (in
``deeprootgen/data_model.py``), and the rotation utilities (in
``deeprootgen/spatial.py``). The following definitions reflect the
implementation supplied for this documentation, not inferred botanical
interpretations.

.. list-table:: Angle and length parameter definitions
   :header-rows: 1
   :widths: 20 49 31

   * - Field concept
     - Implemented model behaviour
     - Implication for the workbook
   * - Starting / insertion angle
     - No direct measured insertion-angle input. Primary roots begin from a
       model displacement direction, then the outer group receives a random
       pitch between -20 and -15 degrees; the inner group receives a pitch
       between 0 and 45 degrees. Yaw positions are spread around the trunk
       with random variation.
     - The measured 2D angles in ``Root angle summary example`` are **not**
       directly the pitch parameters. Compare final generated root directions
       with the observed angle convention instead.
   * - Branching angle
     - No single explicit child-to-parent branching-angle parameter. Secondary
       organs receive rotations including a random yaw from -30 to 240
       degrees, random pitch and roll from 60 to 110 degrees, and additional
       fixed rotations.
     - A measured 3D angle cannot currently be passed as a direct prior on a
       named branching-angle parameter without modifying the model.
   * - Mechanical constraint
     - The ``mechanical_constraints`` input is not applied in the supplied
       simulation methods. Validation uses ``no_root_zone``, iterative pitch
       rotations, and a maximum number of attempts. ``gravitropism`` is a
       separate downward bending operation.
     - Do **not** interpret the workbook's absolute rooting angle as a direct
       prior for ``mechanical_constraints`` in the current implementation.
   * - Segment variance (``root_vary``)
     - Each segment draws two angular perturbations independently from
       ``Uniform(-root_vary, +root_vary)``, used for pitch and yaw. Despite
       the workbook description, this is a uniform range/half-width in
       degrees, **not** a variance statistic or three-axis variance.
     - The available 2D field angles do not directly estimate this
       per-segment perturbation; root-path tracings would help.
   * - Segment length
     - ``init_lengths`` draws a total organ length from a uniform range,
       splits it into ``segments_per_root`` positive, decreasing random
       proportions, and makes their sum equal the sampled organ length.
       Secondary lengths are also adjusted using parent-segment length.
     - Excavated 'length to next branch or recovered end' is not necessarily
       a model segment length or full organ length. Many observed values are
       lower bounds because roots were broken during excavation.
   * - Length reduction (``length_reduction``)
     - Applied by parent/branching order to secondary length intervals,
       not as a fixed reduction from one segment to the next.
     - Use root-order summaries cautiously, distinguishing the observed
       truncated pieces from complete simulated organs.
   * - Inter-branch distance
     - Child attachment currently selects an existing parent segment index
       within a configured fraction of the parent organ. The supplied code
       does not use ``interbranch_distance`` to set physical spacing.
     - The measured Site 1 inter-branch distances cannot yet constrain this
       parameter without implementing a distance-based branching rule.

The ``Root angle summary example`` gives mean first-order angles (positive
downwards) of 17.3, 8.8, -2.2, 31.0 and 15.6 degrees for Sites 1–5,
respectively. These are 2D field estimates; they should not be equated to
3D pitch, roll, yaw or parent-child branching angle without a consistent
coordinate transformation. Because the workbook identifies 3D remeasurement
as time-consuming, first assess sensitivity to angular parameters and whether
existing 2D angles discriminate simulated architectures.

3. How is the location and sampled volume of RLD incorporated?
-------------------------------------------------------------

The ``Data description`` worksheet specifies **14 core locations per site**,
with 10 cm depth increments of **74.22 cm3**. An observation therefore
represents the average RLD of a small **cylindrical sampling volume**, not a
point sample at ``(x, y, z_mid)``. The simulator's voxel table instead
represents an entire rectangular soil cell. Matching the closest voxel alone
would change the observation support and may bias calibration, especially
near sharp spatial gradients.

For an increment of height 10 cm and volume 74.22 cm3, the implied core
radius is approximately **1.537 cm**, assuming a right circular cylinder.
The appropriate forward observation operator is:

.. math::

   \mathrm{RLD}_{\mathrm{core}} =
   \frac{\sum_i L_i(\mathrm{segment}_i\cap\mathrm{core})}
        {\pi r^2 (z_{\mathrm{upper}}-z_{\mathrm{lower}})}.

Here each numerator term is the length of a simulated root segment inside
the specific core cylinder; select the root diameters consistent with the
observational protocol (the workbook's primary grapevine RLD target includes
roots up to approximately **2.5 mm**). The original simulation stores root
diameter in **cm**, so 2.5 mm corresponds to **0.25 cm**.

Direct segment–cylinder intersection is the preferred implementation for
calibration. The previously proposed ``calculate_core_rld`` function performs
this calculation, but it is **not confirmed to exist in the repository**.
Whole-voxel RLD remains useful for visualisation and spatial summaries, but
is not interchangeable with core RLD when the voxel is substantially larger
than the observed core.

There is also a provenance issue: field cores near the subject vine can
intercept roots from neighbouring vines, but their parent vine cannot be
identified from a washed core. The workbook proposes a representative vine
and potentially a 3-by-3 neighbourhood. A single simulated vine should not
be assumed to explain every observed core without testing this assumption.

4. Is the apple calibration input and run script available as a template?
------------------------------------------------------------------------

**Not confirmed from the accessible public repository materials.** DeepRootGen
has a calibration framework, but no specific, verifiable public apple
calibration input file and runnable apple calibration script could be
identified for this FAQ. It would be misleading to claim a template is
available at an unverified path.

The expected observational structure for grapevine fitting is clearer from
the workbook's ``RLD (x, y, z)`` worksheet than from a generic apple
summary-statistics example. A reusable calibration example should demonstrate
how to load that CSV, exclude unsampled increments, transform positive depth
to model ``z`` bounds, select the <=2.5 mm diameter class, calculate predicted
core-volume RLD, and form an objective for each sampled increment. That
spatial calibration integration is still to be provided and tested.

The apple study mentioned in ``possible model parameters`` (for example,
50 segments per root) is not evidence that its calibration files are present
in the repository. If an apple project input and runner are supplied
separately, they can serve as a workflow template, but their observation
schema should not be assumed to support cylindrical core locations.

5. What does a secondary root represent: structural branches or fine roots?
--------------------------------------------------------------------------

**Secondary denotes branching order, not diameter or tissue class.** In the
supplied model, order 1 grows from the trunk/base, order 2 branches from
order 1, and order 3 branches from order 2. A second-order root can be
classified as either ``structural`` or ``fine``. The classification is made
from the organ's **mean simulated diameter** relative to
``fine_root_threshold``. This threshold is an absolute diameter in **cm**,
not a ratio, despite the workbook's description as 'Fine root threshold
ratio'. The workbook's 2.5 mm criterion would correspond to a model
threshold of **0.25 cm** if that observational definition is adopted.

.. list-table:: Which data can inform which parts of the model?
   :header-rows: 1
   :widths: 29 37 34

   * - Dataset
     - Appropriate model target
     - Caution
   * - 2: in-field first-order roots
     - Counts, diameters, lengths and broad insertion directions of primary
       structural roots.
     - Exposed lengths often end at a branch or excavation boundary, not the
       biological root tip.
   * - 3: excavated partial roots
     - Branching orders, structural order-2+ branches, branch counts,
       diameter by order and inter-branch distances.
     - Only surviving structural pieces are traced; measured lengths are
       lower bounds. They do not describe the full fine-root population.
   * - 1: soil coring / WinRHIZO
     - Spatial RLD target and length distribution by root diameter class for
       small roots.
     - Does not provide parent–child topology or complete root-organ length.
   * - 4: RootFly
     - Cross-check diameter/length of coarser fragments, especially
       >2.5 mm and 2.0–2.5 mm.
     - Fragment lengths are limited by the core dimensions.

For example, ``Root size summary example`` reports Site 1 mean start
diameters of **23.0 mm** for order 1 (n=53), **14.3 mm** for order 2
(n=66), and **10.8 mm** for order 3 (n=29). It also shows some second-order
structural roots as large as **31.2 mm**. Thus the large branches measured
on the excavated partial samples are consistent with **secondary structural
roots** in the model; secondary does not mean fine.

A further implementation detail matters for fitting: ``RootOrgan`` currently
classifies a whole generated organ using the mean of its segment diameters,
while field analyses classify roots or root length using a **diameter
threshold per measured fragment/class**. These produce different fine-root
RLD quantities. To compare with WinRHIZO, filter simulated length by the
segment diameter cutoff (and clarify boundary conventions), rather than
relying only on the organ-level ``root_type`` flag. In addition, the
``root_ratio`` parameter is not used in the supplied root-generation methods,
so a structural:fine ratio prior would not currently constrain simulation
outputs without a corresponding model change.

Recommendation
--------------

Use dataset 1 (sampled core increments) as the **spatial RLD fitting target**;
use datasets 2 and 3 chiefly to constrain the **structural branching
framework**; and use WinRHIZO diameter classes together with dataset 4 to
constrain the **small-root diameter distribution** and check the 2.0 versus
2.5 mm cutoff. Before calibration, implement and validate the cylindrical
core observation operator, coordinate alignment to vine row direction, and
consistent diameter filtering. Treat measured excavated lengths as partial
observations rather than exact complete-organ lengths.
