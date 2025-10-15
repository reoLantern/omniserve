# pip uninstall -y qserve_backend
python setup.py clean --all || true
rm -rf build/ *.egg-info

export MAX_JOBS=64  # ninja parallelism

pip install -v --no-build-isolation .
