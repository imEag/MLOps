Vendored internal wheels

Place built internal distribution wheels here (e.g. sovaharmony-<version>.whl).
These are installed in the Docker image after Poetry dependencies.

Update process:
1. Rebuild wheel in source project:
   python -m build --wheel
2. Copy dist/sovaharmony-*.whl into this directory.
3. Rebuild Docker image.
