# Instructions for Replication

To pretrain the deep learning model (required for replication of all the subsequent applications)

- `preprocess_openaitanzania.py`: Pre-process OpenAITanzania training data.

To replicate the analysis on the GiveDirectly (GD) randomized controlled trial in rural Kenya

- `python gd_prepare_download_gsm.py`: Prepare for downloading Google Static Map data.
- `python gd_download_gsm.py`: Download Google Static Map data.
- `python gd_sample_for_annotation.py`: Randomly sample images from the downloaded data for annotation on [Supervisely](https://supervise.ly/). Save annotations in `data/Siaya/Mask/`.
