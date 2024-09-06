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

# install from a local wheel file
pip install --use-wheel --no-index --find-links=dist discretize[test]


echo "Installed discretize version:"
python -c "import discretize; print(discretize.__version__)"