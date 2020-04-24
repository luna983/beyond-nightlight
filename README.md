# Instructions for Replication

To pretrain the deep learning model (required for replication of all the subsequent analyses)

- `preprocess_openaitanzania.py`: Pre-process OpenAITanzania training data.
- `pretrain_oat.py`: Pretrain Mask R-CNN on OpenAITanzania training data.
- `pretrain_pool.py`: Pretrain Mask R-CNN on pooled Google Static Maps images from rural Kenya, peri-urban Tanzania, and rural Mexico. (Before running the script, copy `runs/run_00_PretrainOAT/best_checkpoint.pth.tar` to `runs/run_01_PretrainPool/pretrained_checkpoint.pth.tar`.)

To replicate the analysis in the main text (on the GiveDirectly (GD) randomized controlled trial in rural Kenya)

- `python gd_prepare_download_gsm.py`: Prepare for downloading Google Static Map data.
- `python gd_download_gsm.py`: Download Google Static Map data.
- `python gd_sample_for_annotation.py`: Randomly sample images from the downloaded data for annotation on [Supervisely](https://supervise.ly/). Save annotations in `data/Siaya/Mask/`.
- `python gd_train.py`: Fine tune Mask R-CNN on in-sample annotations in rural Kenya. (Before running the script, copy `runs/run_01_PretrainPool/best_checkpoint.pth.tar` to `runs/run_02_Siaya/pretrained_checkpoint.pth.tar`.)
- `python gd_infer.py`: Run inference to generate predictions on all the images in Siaya.
- `python gd_postprocess.py`: Post-process inference results and collate into a geojson file.

To replicate `fig-prcurve.pdf`:

- `python gd_fig_prcurve_cv.py`: Cross validation on in-sample annotations in rural Kenya. (Before running the script, copy `runs/run_01_PretrainPool/best_checkpoint.pth.tar` to `runs/run_03_SiayaCV0/pretrained_checkpoint.pth.tar`, `runs/run_04_SiayaCV1/pretrained_checkpoint.pth.tar` and `runs/run_05_SiayaCV2/pretrained_checkpoint.pth.tar`.)
- `python gd_fig_prcurve.py`: Generates the raw figure.
