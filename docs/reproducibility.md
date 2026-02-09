# Reproducibility

## Environment
- Python >= 3.9
- Dependencies are listed in `pyproject.toml` and `requirements.txt`.

## Recommended Workflow
- Use a clean virtual environment.
- Install with `pip install -e .` for development or `pip install knot-recognition` for usage.

## Determinism Notes
- Model training uses randomized augmentations and data shuffling.
- For reproducible runs, set global seeds and pin dependency versions.
