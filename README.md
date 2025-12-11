# Coursework 2

[Link to Poster](poster.pdf)

## Table of Contents

1. [Overview](#Overview)
2. [How to Run](#How-to-run)
3. [Background](#Background)
4. [Replicating the Baseline](#Replicating-the-baseline)
5. [New Context](#New-context)
6. [New Dataset](#New-dataset)
7. [Adapted Architecture](#Adapted-architecture)
8. [Model Evaluation](#Model-evaluation)

## Overview

For this coursework, I adapted David John and Ce Zhang's paper: [An attention-based U-Net for detecting deforestation within satellite sensor imagery](https://www.sciencedirect.com/science/article/pii/S0303243422000113#s0020) so that it could perform the task of binary segmentation of Sentinel-2 satellite images for mangrove detection to monitor ecosystem loss over time. I cloned their code and replicated the baseline model performance on their data, before curating a new dataset for the mangrove context, training their models on the new dataset, and then optimizing the models for the specific context. Additionally, I focused on extending the mission of their paper, to identify a lighter-weight, more efficient model that could perform as well as the bulky, expensive state of the art models.

## How to Run

To rerun baseline models, see [Baseline_Model.ipynb](Baseline_Model.ipynb). As the 4-band dataset is too large to fit in github, either download the data using the cell in the notebook, or manually download from [https://zenodo.org/record/4498086#.YMh3GfKSmCU](https://zenodo.org/record/4498086#.YMh3GfKSmCU) and place inside baseline_datasets/4band_AMAZON

To see metrics or rerun models on Mahakam-only mangrove dataset, see [Mangrove_Mahakam_Dataset.ipynb](Mangrove_Mahakam_Dataset.ipynb).

To see metrics or rerun models on combined Mahakam and Tarakan mangrove dataset, see [Mangrove_Combined_Dataset.ipynb](Mangrove_Combined_Dataset.ipynb).

## Background

John and Zhang's paper sets out to implement an Attention U-Net model for binary segmentation of Sentinel-2 satellite images into forested and non-forested pixels, to be used at different timestamped images of the same area of interest to identify and quantify deforestation. The paper also demonstrates how the Attention U-Net, despite its reduced complexity (fewer trainable parameters) and faster training time, is still able to outperform 4 state-of-the-art models: U-Net, Residual U-Net, ResNet50-SegNet and FCN32-VGG16. This makes this paper not only a presentation of a novel approach to deforestation identification, but also important progress in efficiency gains for AI models, demonstrating parameter reduction and model shrinkage without loss of accuracy.

## Replicating the Baseline

### Clone original repository

I successfully cloned the original [GitHub repository](https://github.com/davej23/attention-mechanism-unet/tree/main) for the deforestation Attention U-Net model. The repo contained:

- Model definitions for the five models evaluated in the paper

- Preprocessing scripts for the Amazon forest RGB and 4-band Sentinel-2 datasets

- Training scripts and notebooks used in the original paper

I used the Experiment-Code notebook (which I renamed to baseline_model.ipynb with my slight modifications for environment in my submission) to process the RGB and 4-Band Amazon forest datasets and train, run, and evaluate the models to reproduce the baseline metrics.

### Dependencies and environment setup

I tried to recreate the original environment setup as best I could:

Language / Frameworks

- Python 3.9

- TensorFlow / Keras (TF 2.11+ with tf.keras)

- NumPy, SciPy, scikit-learn

- rasterio, rioxarray, GDAL (for geospatial data)

- matplotlib for visualization

Environment

- Conda environment

- Training was done fully on CPU for this project, no GPUs were used

### Reproduction of Baseline

Before training on my own dataset, I used [Baseline_Model.ipynb](Baseline_Model.ipynb) to train on the RGB and 4band Amazon datasets from the original codebase. All of the code to reproduce the results can be found and executed in the [notebook](Baseline_Model.ipynb). I did not attempt to download or reproduce the Atlantic dataset or results, because I was not intending on demonstrating cross-domain performance for this coursework, although it is an extension I'm interested in for different mangrove regions globally.

My U-Net and Attention U-Net models reached parity with the paper's reported results on the RGB and 4-band Amazon dataset.

## New Context

### Problem and SDG alignment

I chose coastal mangrove loss in Kalimantan, Indonesia, focusing initially on the Mahakam Delta in East Kalimantan and later additional AOIs around Tarakan and other Kalimantan mangrove belts. Mangroves are essential ecosystems as they provide coastal protection from erosion & natural disaster, act as carbon sinks, and support biodiversity by creating a unique habitat for many marine species. But coastal mangroves are deeply at risk. Over 2/3 of global mangrove habitat has already been lost or degraded. Indonesia has the world's largest mangrove area, with the most diversity of species on the planet, but it's at grave risk from palm oil harvesting, aquaculture, and development. From 1994 to 2015, over half of the mangrove forest in the Mahakam Delta in Kalimantan has been converted to aquaculture.

Problem: Mapping mangrove extent and change (loss/gain) at 10 m using Sentinel-2 and Global Mangrove Watch (GMW) labels.

**SDG alignment:**

- SDG 14: Life Below Water: Mangroves provide nursery habitat and stabilize coastal ecosystems.

- SDG 15: Life on Land: Mangroves are forest ecosystems with high biodiversity and carbon storage.

- SDG 13: Climate Action: Mangroves store large amounts of blue carbon; monitoring loss is key to climate mitigation and adaptation strategies.

The same AI approach used for deforestation (semantic segmentation from satellite imagery) is directly applicable to mapping mangrove presence/absence and assessing change between years.

For this coursework, I chose specifically to apply the model to coastal mangroves on the island of Kalimantan in Indonesia.

Compared to the original paper’s task, segmenting mangroves presents new challenges of narrow & fragmented coastal belts, mixed pixels at water-land edges, and noisy labels. These will be the challenges I'll need to tune my model and data pipeline to account for to reach high levels of accuracy.

### Limitations and ethical considerations

Label quality and resolution:

- I used Global Mangrove Watch (GMW) 10 m Sentinel-2 baseline data as ground truth. While high quality, it still has boundary errors and mixed pixels at the land–sea interface.

Misinterpretation risk:

- There’s a risk that imperfect automated maps are used directly for enforcement or blame, without adequate validation or uncertainty estimation.

- Ethical use requires clearly stating accuracy limits, biases (e.g., better performance in wide mangrove belts vs narrow fringes), and the intended scope (monitoring, not legal evidence).

Local context and representation:

- Local communities depend on mangroves for livelihoods; mapping loss without context risks framing historically-rooted cultural practices and means of survival solely as drivers of environmental degradation.

- These tools should be developed and interpreted with input from local communities, researchers, and policy makers to ensure they align with local priorities and knowledge.

### Scalability and sustainability analysis

Scalability:

- The model uses 512×512 Sentinel-2 tiles as inputs and can be applied to any coastal region with S2 coverage and GMW or similar labels.

- The pipeline (Sentinel-2 from Google Earth Engine + GMW + attention U-Net) is directly transferable to other mangrove regions globally.

- The separable Attention U-Net is lightweight (~450k parameters), which makes large-scale tiling and prediction more feasible and sustainable, even on personal computers.

Sustainability:

The approach _is_ sustainable because:

- It relies on openly available data and code.

- It reduces compute and energy cost relative to the heavier baseline U-Nets.

However, running this consistently over decades would require:

- Automated ingestion of new Sentinel-2 data.

- Robust monitoring, versioning of models, and integration with decision-making frameworks.

Which could result in unsustainable practices, energy use, and resource consumption.

## New Dataset

I ended up using two main datasets. My initial dataset was exclusively the Mahakam Delta, with 30 training images, 15 validation images, and 15 test images, which is the exact same parity as the original paper's RGB dataset. This is the dataset for which the metrics on my poster correspond to.

In an attempt to generalize, and see if my conclusions held with more data, I also downloaded corresponding Sentinel-2 images and GMW masks from another delta on the same island, Kalimantan, near Tarakan Island. I combined this with the initial dataset to get 94 training images and 35 validation images from different areas of mangroves to provide a more robust and diverse set of images to my model.

### Contextually appropriate dataset

I curated my mangrove datasets by combining:

Input imagery:

- Sentinel-2 Level-2A surface reflectance

- Filtered AOIs in Kalimantan (Mahakam Delta, Tarakan)

- Cloud filtering and median composites over multi-year windows (2019–2021), using bands B4, B3, B2, B8.

Labels:

- Global Mangrove Watch GMW 10m Sentinel-2 for 2020

This dataset matches the original paper’s setup (Sentinel-2, 10m, binary segmentation) but is tailored to mangroves rather than Amazon forest.

### Data collection process and ethical considerations

Data access:

- I obtained all Sentinel-2 images and GMW labels from Google Earth Engine, and then clipped them, aligned them, and split them into tiles for processing

Ethical considerations:

- All of the data is freely available remote sensing imagery and global masks produced by NGOs, so there’s no private personal data or restrictions to access.

- However, maps of habitat loss could be politically sensitive; care is needed to avoid misinterpretation, especially where ground truth is limited, and local input may not have been taken into account.

### Data preprocessing pipeline

I implemented the following preprocessing pipeline:

1. In Earth Engine:

- Filter Sentinel-2 images by AOI, date range, and CLOUDY_PIXEL_PERCENTAGE.

- Create median composites for each AOI.

- Select B4, B3, B2, B8 bands.

- Export 4-band images at 10m resolution.

2. Getting and aligning Labels:

- Load GMW 2020 mask in Earth Engine.

- Clip to AOI and resample to match the input images.

- Export as binary 0/1 mask.

3. Local preprocessing:

- Convert Sentinel-2 int16 reflectance to uint8 using a fixed scaling.

- Use rasterio to tile both image and mask into 512×512 patches, ensuring alignment.

- Discard tiles where the mask coverage or AOI coverage is incomplete.

- Split into train, validation, and test sets.

4. While processing for the models:

- Normalize image tiles to [0,1].

- For training, create tf.data.Dataset objects from the tiled arrays.

This pipeline mirrors the original paper's setup but with a different AOI and label source.

## Adapted Architecture

### Architectural modifications for new context

I implemented two main architectural adaptations:

1. Separable Attention U-Net (parameter decrease & efficiency)

- Motivation: The original Attention U-Net used Conv2D everywhere, resulting in 2M+ parameters. While the Attention U-Net was already an efficiency gain and much smaller than the other models evaluated in the paper, I wanted to see if I could shrink the models even further without loss of accuracy.

- Change: Replace the Conv2D layers in encoder/decoder blocks with depthwise separable convolutions, while keeping 1×1 convolutions in attention blocks as standard Conv2D.

- Result: Parameter count reduced from \~2M to \~450k, while F1 on the mangrove validation set increased on the Mahakam data set, and decreased only slightly on the larger combined dataset. This is a significant efficiency gain for the same or better performance.

- Regularization and stability (optional variant)

2. To better handle small sample sizes, potential label noise near tidal boundaries, and the smaller parameter size of the model, I experimented with:

- Adding BatchNormalization after convolutions,

- Using LeakyReLU instead of ReLU to reduce dead neurons,

- Applying Dropout in deeper encoder layers for regularization.

This combination is particularly relevant for the mangrove context, where subtle spectral differences and noisy labels can easily lead to overfitting. It also is especially helpful with the separable convolution pattern, as it provides better information to the fewer parameters.

I also experimented with other adaptions like adding X,Y coordinate channels and channel attention blocks, but these did not outperform the simpler separable Attention U-Net on this dataset.

## Model Evaluation

The following tables show the four models I evaluated on the smaller (Mahakam Delta exclusive) dataset, and the larger combined dataset (Mahakam Delta and Tarakan Island).

### Mahakam Only:

| Model                                                                         | Accuracy | Precision | Recall | F1-Score | Parameters |
| ----------------------------------------------------------------------------- | :------: | :-------: | :----: | :------: | :--------: |
| U-Net (Unmodified)                                                            |  0.9488  |  0.9499   | 0.9489 |  0.9494  |    31M+    |
| Attention U-Net (Unmodified)                                                  |  0.9483  |  0.9497   | 0.9483 |  .9490   |    2M+     |
| Optimized Attention U-Net (Batch Normalization, Leaky ReLU, and Dropout)      |  0.9443  |  0.9885   | 0.9442 |  0.9463  |    2M+     |
| 'Mini Model' (Optimized Attention U-Net with Separable Depthwise Convolution) |  0.9499  |  0.9507   | 0.9499 |  0.9503  |    454K    |

From the localized dataset only, we can see that the smallest, most efficient model, was in fact the best model for the problem, achieving the highest metrics of all four of the models evaluated. This shows massive gains in efficiency, and affirms that often for localised problems, bigger and bulkier models can be more wasteful without providing considerable impact.

### Mahakam + Tarakan Dataset:

| Model                                                                         | Accuracy | Precision | Recall | F1-Score | Parameters |
| ----------------------------------------------------------------------------- | :------: | :-------: | :----: | :------: | :--------: |
| U-Net (Unmodified)                                                            |  0.9490  |  0.9539   | 0.9490 |  0.9514  |    31M+    |
| Attention U-Net (Unmodified)                                                  |  0.9472  |  0.9515   | 0.9472 |  .9493   |    2M+     |
| Optimized Attention U-Net (Batch Normalization, Leaky ReLU, and Dropout)      |  0.9426  |  0.9504   | 0.9427 |  0.9465  |    2M+     |
| 'Mini Model' (Optimized Attention U-Net with Separable Depthwise Convolution) |  0.9466  |  0.9507   | 0.9466 |  0.9487  |    454K    |

From the larger combined dataset, we can see that in fact the U-Net and Attention U-Net do outperform the mini model, but only barely, and in fact, the less than 0.003 difference may not justify the nearly 70x more trainable parameters in the U-Net than the mini model.

None of the models trained and evaluated on the mangrove datasets managed to reach the \~0.98 f1-score of the attention U-Net model on the Amazon deforestation dataset. This is likely due to a number of factors:

1. The adjusted model performs well on small channel boundaries, but has trouble classifying thin bands of mangrove, as well as the distinguishing the noisy boundaries between mangrove and aquaculture ponds. As can be seen in the sample prediction images in the poster, noisy boundaries and thin channel-side bands of mangroves can be missed by the model. These shapes are distinct and more complex than the forest shapes, which tend to be larger polygonal masses, and therefore could be easier to identify.

2. The Global Mangrove Watch 10m dataset only reports 95% accuracy, so there are instances in which the model produces correct classifications that diverge from GMW and are marked as error.
