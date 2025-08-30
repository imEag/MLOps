from sovaharmony.pipeline import pipeline as sovaharmony_pipeline
from pathlib import Path
import pandas as pd

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
    datasets,
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
    
def process_data(input_path):
  # validate input path
  path = Path(input_path)
  if not path.exists():
    raise FileNotFoundError(f"Input path does not exist: {input_path}")

  BIOMARCADORES_CE = {
    'name':'original copy',
    'input_path': str(path),
    'layout':{'extension':'.vhdr', 'task':'CE','suffix':'eeg', 'return_type':'filename', 'session':'V0'},
    'args':{'line_freqs':[60]},
    'group_regex':'(.+).{3}',
    'events_to_keep':None,
    'run-label':'restCE',
    'session':'V',
  }
  
  THE_DATASETS=[BIOMARCADORES_CE]
  
  run_pipeline(
    datasets=THE_DATASETS,
    prep=True,
    post=False,
    prepdf=True,
    propdf=False
  )
  run_pipeline(
    datasets=THE_DATASETS,
    prep=False,
    post=True,
    prepdf=False,
    propdf=True
  )
  
  # read the first .feather file present in the folder {input_path}/derivatives/data_columns/IC. Throw an error if the folder or file does not exists.
  ic_folder = path / 'derivatives/data_columns/IC'
  if not ic_folder.exists():
    raise FileNotFoundError(f"IC folder does not exist: {ic_folder}")

  feather_files = list(ic_folder.glob("*.feather"))
  if not feather_files:
    raise FileNotFoundError(f"No .feather files found in: {ic_folder}")

  # Read the first .feather file
  data = pd.read_feather(feather_files[0])

  # remove unwanted columns
  for col in data.columns:
      if col in [
          "database",
          "participant_id",
          "visit",
          "condition",
          "group",
          "sex",
          "age",
          "MM_total",
          "FAS_F",
          "FAS_S",
          "FAS_A",
          "education",
          "education-2",
      ]:
          data.drop(columns=[col], inplace=True)

  return data