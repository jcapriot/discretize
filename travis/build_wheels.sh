#!/bin/bash
set -e -x

#Compile Wheels
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" install -r /io/requirements_dev.txt
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the discretize wheels
for whl in wheelhouse/discretize*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/
done
