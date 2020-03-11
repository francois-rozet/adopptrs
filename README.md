# Automatic Detection Of Photovoltaic Panels Through Remote Sensing

Nowadays, photovoltaic panels are playing an increasingly important role in the global production of electrical energy. Unfortunately, since anyone owning a roof could potentially install PV panels, it is quite hard to assess their geographical deployement and, as a consequence, their impact on the electrical grids.

The goal of this project, named *Automatic Detection Of Photovoltaic Panels Through Remote Sensing* or **ADOPPTRS**, is to provide an easy-to-use tool able to detect photovoltaic panels in high-resolution satellite images.

## Implementation

* [PyTorch](https://pytorch.org/)
* [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

## Training data

For training our models, we used the [Distributed Solar PV Array Location and Extent Data Set for Remote Sensing Object Identification][duke-dataset] provided by [Duke University Energy Initiative](https://energy.duke.edu/).

This dataset contains the geospatial coordinates and border vertices for over `19 000` solar panels across `601` high resolution images from four cities in California.

```bash
wget "https://ndownloader.figshare.com/articles/3385780/versions/3" -O polygons.zip
wget "https://ndownloader.figshare.com/articles/3385828/versions/1" -O Fresno.zip
wget "https://ndownloader.figshare.com/articles/3385789/versions/1" -O Modesto.zip
wget "https://ndownloader.figshare.com/articles/3385807/versions/1" -O Oxnard.zip
wget "https://ndownloader.figshare.com/articles/3385804/versions/1" -O Stockton.zip
mkdir resources
unzip polygons.zip -d resources/polygons/
unzip Fresno.zip -d resources/Fresno/
unzip resources/Modesto.zip -d resources/Modesto/
unzip resources/Oxnard.zip -d resources/Oxnard/
unzip resources/Stockton.zip -d resources/Stockton/
```

[duke-dataset]: https://energy.duke.edu/content/distributed-solar-pv-array-location-and-extent-data-set-remote-sensing-object-identification
