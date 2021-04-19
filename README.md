# Using Satellite Imagery and Deep Learning to Evaluate the Impact of Anti-Poverty Programs

_Luna Yue Huang_

[![DOI](https://zenodo.org/badge/192596578.svg)](https://zenodo.org/badge/latestdoi/192596578)

[__Working Paper__](http://luna-yue-huang.com/assets/pdf/jmp.pdf) | [__Website__](http://luna-yue-huang.com/research-jmp.html) | [__GitHub__](https://github.com/luna983/beyond-nightlight)

----

## Figure Raw Data

| Figure Identifier | Figure No. | CSV File |
| --- | --- | --- |
| fig-schematic | Figure 1 | N/A |
| fig-map | Figure 2 | `fig-map.csv` |
| fig-ate | Figure 3 | `fig-ate.csv` |
| fig-engel | Figure 4 | `fig-engel-a.csv` for Panel a, `fig-engel-b.csv` for Panel b |
| fig-prcurve | ED Figure 1 | `fig-prcurve.csv` |
| fig-chips | ED Figure 2 | N/A |
| fig-colors | ED Figure 3 | N/A |
| fig-engel-housing | ED Figure 4 | `fig-engel-a.csv` for Panel a, `fig-engel-b.csv` for Panel b |
| fig-engel-nonhousing | ED Figure 5 | `fig-engel-a.csv` for Panel a, `fig-engel-b.csv` for Panel b |
| fig-engel-consumption | ED Figure 6 | `fig-engel-a.csv` for Panel a, `fig-engel-b.csv` for Panel b |
| fig-engel-tc-diff | ED Figure 7 | `fig-engel-diff.csv` |
| fig-engel-ei-diff | ED Figure 8 | `fig-engel-diff.csv` |
| fig-mx | ED Figure 9 | `fig-mx.csv` |

## Instructions for Replication

To pretrain the deep learning model (required for replication of all the subsequent analyses)

- `python scripts/preprocess_openaitanzania.py`: Pre-process OpenAITanzania training data. (Before running the script, place raw data in `data/OpenAITanzania/GeoJSON/` and `data/OpenAITanzania/GeoTIFF/`.)
- `python scripts/pretrain_oat.py`: Pretrain Mask R-CNN on OpenAITanzania training data.
- `python scripts/pretrain_pool.py`: Pretrain Mask R-CNN on pooled Google Static Maps images from rural Kenya, peri-urban Tanzania, and rural Mexico. (Before running the script, copy `runs/run_00_PretrainOAT/best_checkpoint.pth.tar` to `runs/run_01_PretrainPool/pretrained_checkpoint.pth.tar`.)

To download nightlight rasters used for comparison in subsequent analyses

- Go to [Google Earth Engine](https://code.earthengine.google.com), run `scripts/download_nightlight.js`: Download and preprocess nightlight rasters in 2019 for Kenya and Mexico. Save rasters in `data/External/Nightlight/`.

To replicate the analysis in the main text (on the GiveDirectly (GD) randomized controlled trial in rural Kenya)

_Note that the replication of some of the analyses requires field data collected in the GiveDirectly trial (placed in the `data/External/GiveDirectly/` folder). These data contain sensitive geolocation information of the trial participants and thus cannot be shared without IRB approval._

- `python scripts/gd_prepare_download_gsm.py`: Prepare for downloading Google Static Map data.
- `python scripts/gd_download_gsm.py`: Download Google Static Map data.
- `python scripts/gd_sample_for_annotation.py`: Randomly sample images from the downloaded data for annotation on [Supervisely](https://supervise.ly/). Save annotations in `data/Siaya/Mask/`.
- `python scripts/gd_train.py`: Fine tune Mask R-CNN on in-sample annotations in rural Kenya. (Before running the script, copy `runs/run_01_PretrainPool/best_checkpoint.pth.tar` to `runs/run_02_Siaya/pretrained_checkpoint.pth.tar`.)
- `python scripts/gd_infer.py`: Run inference to generate predictions on all the images in Siaya.
- `python scripts/gd_polygonize.py`: Collate the inference results into a geojson / csv file.
- `python scripts/gd_postprocess.py`: Post-process the inference results csv file.
- `python scripts/gd_merge.py --resolution 0.005` and `python scripts/gd_merge.py --resolution 0.001 --placebo 100  --eligible-only`: Merge the satellite observations with field survey data.

To replicate `fig-colors` (roof color K-means clustering result)

- `python gd_fig_colors`: Generate the raw figures.

To replicate `fig-prcurve` (Precision-Recall curve)

- `python scripts/gd_fig_prcurve_cv.py`: Conduct cross validation on in-sample annotations in rural Kenya. (Before running the script, copy `runs/run_01_PretrainPool/best_checkpoint.pth.tar` to `runs/run_03_SiayaCV0/pretrained_checkpoint.pth.tar`, `runs/run_04_SiayaCV1/pretrained_checkpoint.pth.tar` and `runs/run_05_SiayaCV2/pretrained_checkpoint.pth.tar`.)
- `python scripts/gd_fig_prcurve.py`: Generate the raw figure.

To replicate `fig-schematic` and `fig-chips` (schematics and randomly sampled images/predictions)

- `python scripts/gd_fig_schematic.py`: Sample images and predictions.

To replicate `fig-map` (map of treatment and outcome variables)

- `python scripts/gd_fig_map.py > logs/gd_fig_map.log`: Generate the raw figures.

To replicate `fig-ate` (average treatment effect estimation)

- `Rscript scripts/gd_fig_ate.R > logs/gd_fig_ate.log`: Generate raw figures.

To replicate `fig-engel`, `fig-engel-housing`, `fig-engel-nonhousing`, `fig-engel-consumption`, `fig-engel-tc-diff`, `fig-engel-ei-diff` (Engel curves)

- `python scripts/gd_fig_engel.py > logs/gd_fig_engel.log`: Generate raw figures.

To replicate the analysis in the appendix (in rural Mexico)

- `python scripts/mx_prepare_download_gsm.py`: Prepare for downloading Google Static Map data. (Before running the script, download the raw data for the [Population and Housing Census 2010](https://www.inegi.org.mx/programas/ccpv/2010/default.html) on the official INEGI website. Convert the `.dbf` file to a `.csv` file and place it in the `data/External/MexicoCPV/` folder.)
- `python scripts/mx_download_gsm.py`: Download Google Static Map data.
- `python scripts/mx_sample_for_annotation.py`: Randomly sample images from the downloaded data for annotation on [Supervisely](https://supervise.ly/). Save annotations in `data/Mexico/Mask/`.
- `python scripts/mx_infer.py`: Run inference to generate predictions on all the images in the sample.
- `python scripts/mx_postprocess.py`: Post-process inference results and collate into a geojson file.

To replicate `fig-mx` (appendix figures showing validation results in Mexico)

- `python scripts/mx_fig_validate.py`: Generate raw figures.
