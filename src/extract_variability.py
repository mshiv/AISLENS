from modules.data_preprocess import detrend, deseasonalize
from modules.seasonality import extract_seasonality
from modules.draft_dependence import dedraft

def preprocess_model_data():
    # Load model data
    model_file = "data/external/basalmelt_model_data.nc"
    model_data = load_dataset(model_file)

    # Detrend and deseasonalize
    detrended_data = detrend(model_data)
    deseasonalized_data, seasonality_signal = deseasonalize(detrended_data)

    # Save seasonal signal
    save_dataset(seasonality_signal, "data/interim/basalmelt_seasonality_model_data.nc")

    # Dedraft the deseasonalized data
    dedrafted_data = dedraft(deseasonalized_data)

    # Save variability signal
    save_dataset(dedrafted_data, "data/interim/basalmelt_variability_model_data.nc")