# Simple hand gesture detection

### Requiers
* python 3.6
* opencv
* keras
* tensorflow

### How to use
`python detection.py`

### Additional scripts
* `filtering.py` - script breaking down data into learning and validation datasets. It assumes that `data/validation` is exact copy of `data/learning`
* `network.py` - script used for learning process. CNN structure is defined here.
* `video_sampling.py` - scipt used for gathering data. Hit `space` to start recording frames.