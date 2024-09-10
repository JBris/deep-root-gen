![DeepRootGen](docs/logos/deep_root_gen.png)

# DeepRootGen

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/JBris/deep-root-gen/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/JBris/deep-root-gen/actions/workflows/tests.yaml)
[![Documentation](https://github.com/JBris/deep-root-gen/actions/workflows/docs.yaml/badge.svg?branch=main)](https://github.com/JBris/deep-root-gen/actions/workflows/docs.yaml)
[![Build](https://github.com/JBris/deep-root-gen/actions/workflows/docker-build.yaml/badge.svg?branch=main)](https://github.com/JBris/deep-root-gen/actions/workflows/docker-build.yaml)

Website: [DeepRootGen](https://jbris.github.io/deep-root-gen/)

*A simulation model for the digital reconstruction of 3D root system architectures. Integrated with a simulation-based inference generative deep learning model.*

# Table of contents

- [DeepRootGen](#deeprootgen)
- [Table of contents](#table-of-contents)
- [Introduction](#introduction)
- [Data Schema](#data-schema)
- [Contacts](#contacts)
  
# Introduction

This repository contains an implementation of a simulation model for generating a synthetic root system architecture, for the estimation of root parameters as informed by observational data collected from the field.

# Data Schema

The primary purpose of DeepRootGen is to output a synthetic root system into a tabular data format. The tabular data are composed of several columns:

| Column       | Type       | Description                                                                                                                                                                     |
| ------------ | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| id           | Discrete   | A unique ID for each row.                                                                                                                                                       |
| plant_id     | Discrete   | A unique ID for each plant.                                                                                                                                                     |
| organ_id     | Discrete   | A unique ID for each root.                                                                                                                                                      |
| order        | Discrete   | The plant order. A first order root grows from the plant base,  second order roots emerge from first order roots, third order  roots emerge from second order roots, and so on. |
| root_type    | Discrete   | The root type classification. Can be one of 1 = Structural Root or 2 = Fine Root.                                                                                               |
| segment_rank | Discrete   | The rank number for each root segment. A small rank refers to  segments that are close to the root base, while a large rank  refers to roots that are near the root apex.       |
| parent       | Discrete   | The parent organ of the root segment. The parent node of the organ within the GroIMP graph. Used by the XEG reader to  recursively import the synthetic root data.              |
| coordinates  | Continuous | The combined 3D coordinates (x, y, and z) of each root segment.                                                                                                                 |
| diameter     | Continuous | The root segment diameter.                                                                                                                                                      |
| length       | Continuous | The root segment length.                                                                                                                                                        |
| x            | Continuous | The x coordinate of the root segment.                                                                                                                                           |
| y            | Continuous | The y coordinate of the root segment.                                                                                                                                           |
| z            | Continuous | The z coordinate of the root segment.                                                                                                                                           |

# Contacts

- DeepRootGen Developer: James Bristow 
- DeepRootGen Project Supervisor: Junqi Zhu
- Crop System Modeller: Xiumei Yang 
