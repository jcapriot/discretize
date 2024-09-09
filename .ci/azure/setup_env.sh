#!/bin/bash
set -ex #echo on and exit if any line fails

# TF_BUILD is set to True on azure pipelines.
is_azure=$(echo "${TF_BUILD:-false}" | tr '[:upper:]' '[:lower:]')
do_doc=$(echo "${DOC_BUILD:-false}" | tr '[:upper:]' '[:lower:]')

if ${is_azure}
then
  if ${do_doc}
  then
    .ci/setup_headless_display.sh
  fi
fi

echo "python="$PYTHON_VERSION

pip install packaging
wheel_file=$(python .ci/azure/get_wheel.py dist)

# install from the built wheel file
pip install $wheel_file[test]


echo "Installed discretize version:"
python -c "from importlib.metadata import version; print(version('discretize'))"