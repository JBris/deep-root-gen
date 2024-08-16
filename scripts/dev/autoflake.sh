#!/usr/bin/env bash

CMD="autoflake --in-place --remove-duplicate-keys --remove-unused-variables --recursive"
$CMD deeprootgen/
$CMD tests/
