"""Test / example flow for running sovaharmony.pipeline.

This module is placed under src/flows so you can import its callable like:

    from src.flows.test_preprocessing import run_pipeline

and invoke run_pipeline() inside other parts of the backend without triggering
the heavy processing at import time.
"""

# Local dev fallback: if sovaharmony isn't installed (only vendored wheel present),
# add the wheel to sys.path so imports work outside the Docker image.
try:  # pragma: no cover - simple import guard
    from sovaharmony.pipeline import pipeline as sovaharmony_pipeline
except ModuleNotFoundError:  # running in host env without pip-installed wheel
    import sys, pathlib
    vendor_dir = pathlib.Path(__file__).resolve().parents[2] / "vendor"
    # Prefer unpacked source tree if present (vendor/sovaharmony/sovaharmony)
    source_pkg = vendor_dir / "sovaharmony" / "sovaharmony"
    if source_pkg.exists():
        sys.path.insert(0, str(source_pkg.parent))  # add vendor/sovaharmony
    # Fallback: insert wheel archive if available
    for whl in sorted(vendor_dir.glob("sovaharmony-*.whl")):
        if str(whl) not in sys.path:
            sys.path.insert(0, str(whl))
        break
    try:
        from sovaharmony.pipeline import pipeline as sovaharmony_pipeline
    except ModuleNotFoundError as e:  # give clearer hint
        raise ModuleNotFoundError(
            "sovaharmony not available. Install the wheel with 'poetry run pip install vendor/sovaharmony-<version>.whl' or rebuild the Docker image."  # noqa: E501
        ) from e


BIOMARCADORES_CE = {
    'name':'original copy',
    'input_path':r'/Users/imeag/Documents/udea/trabajoDeGrado/MLOps/backend/data/no processed/database_bids',
    'layout':{'extension':'.vhdr', 'task':'CE','suffix':'eeg', 'return_type':'filename', 'session':'V0'},
    'args':{'line_freqs':[60]},
    'group_regex':'(.+).{3}',
    'events_to_keep':None,
    'run-label':'restCE',
    'session':'V',
}

THE_DATASETS=[BIOMARCADORES_CE]
L_FREQ = 1
H_FREQ = 50
epoch_length = 5

spatial=['58x25']

metrics={'power':{'fit_params':False,'norm':'False','demographic':False},
         'osc': {'fit_params':False,'norm':'False','demographic': False},
         'ape': {'fit_params':True,'norm':'False' ,'demographic': False},
         'ape': {'fit_params':False,'norm':'False','demographic': False}
         }

bands ={'Delta':(1.5,6),
        'Theta':(6,8.5),
        'Alpha-1':(8.5,10.5),
        'Alpha-2':(10.5,12.5),
        'Beta1':(12.5,18.5),
        'Beta2':(18.5,21),
        'Beta3':(21,30),
        'Gamma':(30,45)
        }

def run_pipeline(
    datasets=THE_DATASETS,
    l_freq=L_FREQ,
    h_freq=H_FREQ,
    epoch_len=epoch_length,
    prep=False,
    post=True,
    portables=False,
    prepdf=False,
    propdf=True,
    spatial_matrix=spatial,
    metrics_cfg=metrics,
    ic=True,
    sensors=False,
    roi=False,
    overwrite=True,
    bands_cfg=bands,
):
    """Execute sovaharmony pipeline with default test configuration.

    Parameters can be overridden when calling for experimentation.
    Returns whatever the underlying sovaharmony_pipeline returns.
    """
    return sovaharmony_pipeline(
        datasets,
        L_FREQ=l_freq,
        H_FREQ=h_freq,
        epoch_length=epoch_len,
        prep=prep,
        post=post,
        portables=portables,
        prepdf=prepdf,
        propdf=propdf,
        spatial_matrix=spatial_matrix,
        metrics=metrics_cfg,
        IC=ic,
        Sensors=sensors,
        roi=roi,
        OVERWRITE=overwrite,
        bands=bands_cfg,
    )


__all__ = ["run_pipeline"]


if __name__ == "__main__":  # Optional manual run
    import inspect

    print("Signature:", inspect.signature(sovaharmony_pipeline))
    print("Module:", sovaharmony_pipeline.__module__)
    run_pipeline()