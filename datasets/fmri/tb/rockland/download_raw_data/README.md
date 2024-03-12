# Downloading raw Rockland data

The Rockland data set is open access and does not need any form of registration.
For more details see: http://fcon_1000.projects.nitrc.org/indi/enhanced/neurodata.html
We use the downloader script for S3.
Note that we use the second version of this script.

TODO:
subject IDs are at: H:\neuro-dynamic-covariance\datasets\task\rockland\preprocessing\raw\
This script always downloads defiative informations (e.g. physio) that we do not need.
This is optional with a -d flag that we don't specify, but they are still downloaded anyway...

## Download subset

We are only interested in the first baseline visit (coded BAS1).
There are 1495 participants in total for this visit.
Furthermore, we select participants between 18 and 35 years old.

## Checkerboard data

### Checkerboard 645 TR

```
python datasets/task/rockland/download_raw_data/download_rockland_raw_bids.py --out_dir /mnt/Data/neuro-dynamic-covariance/datasets/task/rockland/CHECKERBOARD645/raw -e CHECKERBOARD645 -v BAS1 --greater_than 17 --less_than 36
```

### Checkerboard 1400 TR

```
python datasets/task/rockland/download_raw_data/download_rockland_raw_bids.py --out_dir /mnt/Data/neuro-dynamic-covariance/datasets/task/rockland/CHECKERBOARD1400/raw -e CHECKERBOARD1400 -v BAS1 --greater_than 17 --less_than 36
```
