Run the model independently
===========================

DeepRootGen's root-system simulator can run directly in Python without
starting the Docker Compose stack, Dash, Prefect, MLflow, or any database.
Use this mode for testing root architecture, exploring model parameters,
and analysing simulation output in a notebook.

Install the Python package
--------------------------

Clone the repository and install its Python dependencies with Poetry:

.. code-block:: console

   git clone https://github.com/Plant-Food-Research-Open/deep-root-gen.git
   cd deep-root-gen
   poetry install

Use a Python version supported by the repository's ``pyproject.toml``.
You can now open an interactive session with ``poetry run python``, or use a
Jupyter notebook with the same Poetry environment. No application services
need to be running.

Generate a root system
----------------------

All simulation parameters are specified directly in Python. Paste the entire
example into a notebook cell, or run it in an interactive Python session.
It produces root node and edge tables and an interactive 3D visualisation.

.. code-block:: python

   from pathlib import Path

   from deeprootgen.data_model import RootSimulationModel
   from deeprootgen.model import RootSystemSimulation

   # Define the model parameters in Python.
   params = RootSimulationModel(
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
       gravitropism=0.5,
       t=1,
       enable_soil=False,
       soil_layer_height=25,
       soil_n_layers=8,
       soil_layer_width=25,
       soil_n_cols=4,
   )

   # Run the root-system model.
   simulation = RootSystemSimulation(random_seed=params.random_seed)
   simulation.run(params)

   # Extract the graph as dataframes.
   node_df, edge_df = simulation.G.as_df()
   print(f"Root nodes: {len(node_df)}")
   print(f"Root edges: {len(edge_df)}")
   print(node_df.head())

   # Save data and an interactive Plotly figure.
   output_dir = Path("outputs/standalone")
   output_dir.mkdir(parents=True, exist_ok=True)

   node_df.to_csv(output_dir / "root_nodes.csv", index=False)
   edge_df.to_csv(output_dir / "root_edges.csv", index=False)

   fig = simulation.init_fig(params)
   fig = simulation.plot_root_system(fig, node_df)
   fig.write_html(str(output_dir / "root_system.html"))
   fig.show()  # Displays inline in Jupyter or a compatible environment.

The output ``root_nodes.csv`` records segment endpoints, parent IDs, diameters,
lengths, root orders and root types. ``root_edges.csv`` records the graph's
parent-child links. ``root_system.html`` can be opened in a web browser.

.. note::

   The example uses one random seed, so runs with the same model version and
   dependency versions can reproduce the same stochastic construction. The
   current ``RootSystemSimulation.run()`` implementation does not invoke
   ``grow()``, so ``t`` does not trigger extra growth in that implementation.

Optional: aggregate root data by soil voxel
-------------------------------------------

If your local DeepRootGen version includes
``deeprootgen.statistics.root_grid_aggregation``, you can also obtain spatial
root length density, fine- and structural-root biomass, and per-plant totals.
Continue in the same notebook or Python session after running the example above:

.. code-block:: python

   import numpy as np

   from deeprootgen.statistics.root_grid_aggregation import aggregate_root_grid

   soil_df = simulation.soil.create_soil_grid(
       soil_layer_height=params.soil_layer_height,
       soil_n_layers=params.soil_n_layers,
       soil_layer_width=params.soil_layer_width,
       soil_n_cols=params.soil_n_cols,
   )

   # The original Soil class uses y for depth, while roots use z.
   # Swap only when that original convention is present.
   expected_depth = params.soil_layer_height * params.soil_n_layers
   expected_width = params.soil_layer_width * params.soil_n_cols

   if (
       np.isclose(soil_df["y"].min(), -expected_depth)
       and np.isclose(soil_df["y"].max(), 0)
       and not np.isclose(expected_depth, expected_width)
   ):
       soil_df = soil_df.rename(columns={"y": "z", "z": "y"})

   assert np.isclose(soil_df["z"].min(), -expected_depth)
   assert np.isclose(soil_df["z"].max(), 0)

   grid_df, totals_df = aggregate_root_grid(
       node_df=node_df,
       soil_df=soil_df,
       length_source="recorded",
   )

   print(f"Soil voxels: {len(grid_df)}")
   print(f"Occupied voxels: {(grid_df['root_length_cm'] > 0).sum()}")
   print(totals_df)

   grid_df.to_csv(output_dir / "root_grid_aggregates.csv", index=False)
   totals_df.to_csv(output_dir / "root_biomass_totals.csv", index=False)

This optional step uses the same simulated graph and does not require any
application services. The axis check assumes that soil width and depth differ;
for production use, verify the complete expected ``x``, ``y`` and ``z`` bounds.

Check segment geometry before spatial calibration
-------------------------------------------------

Spatial calibration assumes that the recorded length of a segment is consistent
with the distance between its child endpoint and parent endpoint. You can
check this directly:

.. code-block:: python

   import numpy as np

   parents = node_df.set_index("node_id")
   segments = node_df.loc[
       (node_df["order"] > 0)
       & (~node_df["invalid_root"].fillna(False).astype(bool))
       & node_df["parent_id"].isin(parents.index)
   ].copy()

   parent_xyz = parents.loc[
       segments["parent_id"], ["x", "y", "z"]
   ].to_numpy(dtype=float)
   child_xyz = segments[["x", "y", "z"]].to_numpy(dtype=float)

   geometric = np.linalg.norm(child_xyz - parent_xyz, axis=1)
   recorded = segments["length"].to_numpy(dtype=float)

   print("Maximum length discrepancy (cm):",
         np.max(np.abs(geometric - recorded)))

   np.testing.assert_allclose(recorded, geometric, rtol=1e-6, atol=1e-8)

Keep the data and figure for each simulation together, and record the
DeepRootGen code revision when comparing results across versions.
