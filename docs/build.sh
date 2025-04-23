#!/bin/bash

export TMPDIR=$HOME/scratch
export SPACKDIR=$HOME/data/spack-env
export SPACKENV=mali

export YAML=$PWD/env.yaml

set -e

# load system modules here if you need to, e.g.:
# module purge
# module load gcc

if [ -d $SPACKDIR/develop ]; then
  # update to the latest version of develop
  cd $SPACKDIR/develop
  git fetch origin
  git reset --hard origin/develop
else
  git clone -b develop https://github.com/E3SM-Project/spack $SPACKDIR/develop
  #git clone -b develop git@github.com:E3SM-Project/spack.git $SPACKDIR/develop
  cp $PWD/config.yaml develop/etc/spack/defaults
  cp $PWD/modules.yaml develop/etc/spack/defaults
  cd $SPACKDIR/develop
fi
source share/spack/setup-env.sh
spack env remove -y $SPACKENV &> /dev/null && \
  echo "recreating environment: $SPACKENV" || \
  echo "creating new environment: $SPACKENV"
spack env create $SPACKENV $YAML
spack env activate $SPACKENV
spack install
spack config add modules:prefix_inspections:lib:[LD_LIBRARY_PATH]
spack config add modules:prefix_inspections:lib64:[LD_LIBRARY_PATH]
spack module lmod refresh --delete-tree
