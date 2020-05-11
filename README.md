# Automatic Detection Of Photovoltaic Panels Through Remote Sensing

Nowadays, photovoltaic panels are playing an increasingly important role in the global production of electrical energy. Unfortunately, since anyone owning a roof could potentially install PV panels, it is quite hard to assess their geographical deployement and, as a consequence, their impact on the electrical grids.

Therefore, this project, named *Automatic Detection Of Photovoltaic Panels Through Remote Sensing* or **ADOPPTRS**, aims to detect photovoltaic panels in high-resolution satellite images.

More specifically, the goal is to detect, as accurately as possible, photovoltaic panels in the [WalOnMap][walonmap] orthorectified images in the [Province of Liège](resources/walonmap/liege_province.geojson).

> All the photovoltaic installations that have been detected, can be visualized at [francois-rozet.github.io/adopptrs](https://francois-rozet.github.io/adopptrs/).

## Implementation

The [PyTorch](https://pytorch.org/) library has been used to implement and train several neural networks [models](python/models.py) one of which is the well known [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

> For a short description of the arguments of the scripts (`train.py`, `walonmap.py`, etc.), use `--help`.

### Dependencies

If you wish to run the scripts or the [Jupyter](https://jupyter.org/) notebook(s), you will need to install several `Python` packages including `jupyter`, `torch`, `torchvision`, `opencv`, `matplotlib` and their dependencies.

To do so safely, one should create a new environement :

```bash
virtualenv ~/adopptrs -p python3
source ~/adopptrs/bin/activate
pip3 install -r requirements.txt -y
```

or with `Anaconda`

```bash
conda env create -f environment.yml
```

### Networks

The neural networks that have been implemented (cf. [`models.py`](python/models.py)) are *U-Net*, *SegNet* and [*Multi-Task*](https://arxiv.org/abs/1709.05932) versions of them.

The legacy networks are trained with a *Dice loss* while the multi-task ones are trained with a *Multi-Task loss* (cf. [`criterions.py`](python/criterions.py)).

### Augmentation

During training, the dataset is *augmented*, meaning that each image undergoes a different random transformation at each epoch. The transformation is a combination of *rotations* (90°, 180° or 270°), *flips* (horizontal or vertical), *brightness* alteration, *contrast* alteration, *saturation* alteration, *blurring*, *smoothing*, *sharpening*, etc.

This improves greatly the *robustness* of the networks.

### Reproductibility

In order to produce the networks and plots that are presented in the [notebooks](notebooks/), the script [`train.py`](python/train.py) was called (respectively) with the following arguments :

```bash
python train.py -m unet -n unet -optim adam -o unet.txt -stat unet.csv
python train.py -m segnet -n segnet -optim sgd -o segnet.txt -stat segnet.csv
python train.py -m unet -multitask -n multiunet -optim adam -o multiunet.txt -stat multiunet.csv
python train.py -m segnet -multitask -n multisegnet -optim sgd -o multisegnet.txt -stat multisegnet.csv
```

However, for the network that was selected for [fine-tuning](notebooks/tuning.ipynb), the arguments were

```bash
python train.py -m unet -multitask -n multiunet_x2 -e 50 -s 450 460 -optim adam -scale 2 -o multiunet_x2.txt -stat multiunet_x2.csv
```

Then, the fine-tuned model was applied to the [Province of Liège](resources/walonmap/liege_province.geojson) with

```bash
python walonmap.py -m multiunet_x2_029_20.pth -p liege_province.geojson -o liege_province_via.json
```

Finally, the resulting `liege_province_via.json` file was *"summarized"* using

```bash
python summarize.py -i liege_province_via.json -o liege_province.csv
```

which produced the [`liege_province.csv`](docs/resources/csv/liege_province.csv) file.

> For readability purposes, the files' paths are not necessarily correct.

## Training data

For training our models, we used the [Distributed Solar PV Array Location and Extent Data Set for Remote Sensing Object Identification][duke-dataset] provided by [Duke University Energy Initiative](https://energy.duke.edu/).

This dataset contains the geospatial coordinates and border vertices for over `19 000` solar panels across `601` high resolution images from four cities in California.

```bash
wget "https://ndownloader.figshare.com/articles/3385780/versions/3" -O polygons.zip
wget "https://ndownloader.figshare.com/articles/3385828/versions/1" -O Fresno.zip
wget "https://ndownloader.figshare.com/articles/3385789/versions/1" -O Modesto.zip
wget "https://ndownloader.figshare.com/articles/3385807/versions/1" -O Oxnard.zip
wget "https://ndownloader.figshare.com/articles/3385804/versions/1" -O Stockton.zip
mkdir -p resources/california/
unzip polygons.zip -d resources/california/
unzip Fresno.zip -d resources/california/
unzip Modesto.zip -d resources/california/
unzip Oxnard.zip -d resources/california/
unzip Stockton.zip -d resources/california/
rm *.zip resources/california/*.xml # optionally
```

Afterwards, the file `SolarArrayPolygons.json` has to be converted to the [VGG Image Annotator][via] format.

```bash
python3 python/dataset.py --output products/json/california.json --path resources/california/
```

[walonmap]: https://geoportail.wallonie.be/walonmap
[duke-dataset]: https://energy.duke.edu/content/distributed-solar-pv-array-location-and-extent-data-set-remote-sensing-object-identification
[via]: http://www.robots.ox.ac.uk/~vgg/software/via/
