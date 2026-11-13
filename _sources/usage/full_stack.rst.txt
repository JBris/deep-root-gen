Running the full application with Docker
========================================

DeepRootGen uses Docker Compose to run its interactive application and the
services supporting simulation, calibration and experiment tracking. You do
**not** need Docker to run the simulation model by itself; see :doc:`standalone`.

Prerequisites
-------------

Install Git, Docker Engine or Docker Desktop, and the Docker Compose plugin.
Ensure the machine has sufficient memory and storage for multiple services.

Clone and configure
-------------------

.. code-block:: console

   git clone https://github.com/Plant-Food-Research-Open/deep-root-gen.git
   cd deep-root-gen
   docker compose -f docker-compose.yaml config --services
   docker compose -f docker-compose.yaml config --variables

Review the values in ``.env`` and the Compose file. In particular, verify
image version, database connection settings, credentials, host names, object
storage settings, and exposed ports. Keep secrets out of published files and
replace any development credentials before exposing the stack externally.

Start the main application
--------------------------

.. code-block:: console

   docker compose -f docker-compose.yaml up -d
   docker compose -f docker-compose.yaml ps

The stack includes the Dash interface, Prefect workflow orchestration, MLflow
experiment tracking, PostgreSQL, MinIO object storage and ArangoDB graph
storage. To determine the actual host ports for an installation, use
``docker compose -f docker-compose.yaml ps``; do not assume that container
ports equal host ports.

Model deployment services (optional)
------------------------------------

The repository's ``docker-compose.model-deploy.yaml`` defines additional
BentoML services. These are **optional** and use the same Docker network and
back-end services as the primary Compose file.

======================= ==========
Service                 Host port
======================= ==========
Optimisation            5001
Sensitivity analysis    5002
ABC                     5003
SNPE                    5004
Surrogate modelling     5005
======================= ==========

Run both configurations together:

.. code-block:: console

   docker compose -f docker-compose.yaml -f docker-compose.model-deploy.yaml up -d
   docker compose -f docker-compose.yaml -f docker-compose.model-deploy.yaml ps

The deployed endpoints and readiness depend on their associated BentoML
services and their dependencies. A container starting is not equivalent to a
model being ready to receive requests.

Logs and shutdown
-----------------

.. code-block:: console

   docker compose -f docker-compose.yaml logs -f
   docker compose -f docker-compose.yaml down

For an installation started with both files, supply both files when stopping:

.. code-block:: console

   docker compose -f docker-compose.yaml -f docker-compose.model-deploy.yaml down

Do not add ``-v`` to ``down`` unless deliberately deleting persistent volumes.

Troubleshooting
---------------

* Run ``docker compose -f docker-compose.yaml config`` to validate variable
  expansion and the resulting service definitions.
* Run ``docker compose -f docker-compose.yaml ps`` and ``logs --tail=100`` to
  diagnose containers that fail or restart.
* Confirm that all requested host ports are available.
* Confirm that MLflow and BentoML can reach the relevant storage endpoints
  using *container* network names rather than ``localhost``.
* Use :doc:`standalone` if only a local root architecture simulation is
  needed.
