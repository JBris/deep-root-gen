#!/usr/bin/env bash

###################################################################
# Constants
###################################################################

OUT_DIR=html

###################################################################
# Main
###################################################################

rm -rf $OUT_DIR
cd docs
make $OUT_DIR
mv build/${OUT_DIR} ..