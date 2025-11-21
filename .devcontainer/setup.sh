#!/usr/bin/env bash
set -e

echo "=== Installing system dependencies ==="
sudo apt-get update
sudo apt-get install -y libblas-dev liblapack-dev gfortran libcurl4-openssl-dev libssl-dev libxml2-dev

echo "=== Installing R packages for LEfSe ==="
R -q -e "install.packages(c('survival','mvtnorm','modeltools','coin','MASS','splines','stats4'), repos='https://cloud.r-project.org/')"

echo "=== Installing Python packages ==="
pip install --upgrade pip wheel setuptools
pip install numpy scipy matplotlib cython biom-format rpy2
