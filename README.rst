Vehicle Detection and Tracking
==============================

This project was created to practise the use of a pre-trained vehicle detection
model similar to `this project at pyimagesearch
<https://www.pyimagesearch.com/2019/12/02/opencv-vehicle-detection-tracking-and-speed-estimation/>`_.

Data
----

Data used in this project are pre-recorded videos of cars driving by on a road
or in a parking lot. Some videos have moving cameras, which may not work ideally
with this model. Data is stored in the JSON format, containing the name, date
and URL.

API
---

Pre-trained models from the model zoo of `Facebook AI Research's Detectron2
<https://github.com/facebookresearch/detectron2>`_ were used in this project.
More specifically, a converted copy of MSRA's ResNet-101 model was used, which
is titled `R-101.pkl
<https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl>`_.

Team
----

This project is a collaboration between

1) `Anukaw100 <https://github.com/Anukaw100>`_, who is responsible for
visualizing the data;
2) `ss-raicangu <https://github.com/ss-raicangu>`_, who is responsible for
tracking the vehicles;
3) `RuotongX <https://github.com/RuotongX>`_, who is responsible for configuring
and running the model.

TODOs
-----

Requires the following files and directories:

1) LICENSE,
2) MANIFEST.in,
3) Makefile,
4) setup.py,
5) output/,
6) config/,
7) docs/.

Refer to `sample module repository by navdeep-G
<https://github.com/navdeep-G/samplemod>`_ and `Structury Your Project at
python-guide.org <https://docs.python-guide.org/writing/structure/>`_.
