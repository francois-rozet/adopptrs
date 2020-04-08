# Automatic Detection Of Photovoltaic Panels Through Remote Sensing

Nowadays, photovoltaic panels are playing an increasingly important role in the global production of electrical energy. Unfortunately, since anyone owning a roof could potentially install PV panels, it is quite hard to assess their geographical deployement and, as a consequence, their impact on the electrical grids.

The goal of this project, named *Automatic Detection Of Photovoltaic Panels Through Remote Sensing* or **ADOPPTRS**, is to provide an easy-to-use tool able to detect photovoltaic panels in high-resolution satellite images.

## Implementation

The [PyTorch](https://pytorch.org/) library has been used to implement and train several neural networks [models](python/models.py) one of which is the well known [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

### Notebooks

If you wish to run the [Jupyter](https://jupyter.org/) notebook(s), you will need to install several `Python` packages including `jupyter`, `torch`, `torchvision`, `opencv`, `matplotlib` and their dependencies.

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

[duke-dataset]: https://energy.duke.edu/content/distributed-solar-pv-array-location-and-extent-data-set-remote-sensing-object-identification
[via]: http://www.robots.ox.ac.uk/~vgg/software/via/
