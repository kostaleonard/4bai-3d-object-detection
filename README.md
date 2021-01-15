# 3D Object Detection for Autonomous Vehicles
## A FourthBrain AI Project

Authors:
* Leo Kosta: [Leo's github](https://github.com/kostaleonard)
* Ashish Mahashabde: [Ashish's github](https://github.com/amahashabde)

## Project Description

In order to effectively maneuver, autonomous vehicles rely on the ability to accurately estimate bounding boxes of various objects, including other vehicles. Companies like Lyft have recently experimented with using 3D bounding boxes, which could allow autonomous vehicles to have richer positioning information and make better predictions for maneuvering. By reimplementing recent methods and experimenting with new approaches, we want to improve on the state-of-the-art in bounding volume prediction in the self-driving car domain.

## Approach

Based on the literature we've reviewed (see [References](#references)), we used a 2D object detector to produce priors that help a 3D object detector draw accurate 3D bounding boxes. The 2D object detector draws a rectangular (2D) bounding box around the cars, and the 3D object detector uses those boxes to infer 6-sided (3D) bounding volumes. We use **only monocular RGB camera data**, with no depth information. 

We know it can be a nightmare to use code you haven't personally written. Our goal is that you can clone this repository and run the model in a single command. When you're ready to find out how it works, you'll find well-written, well-documented, well-tested code.

## Run Instructions

Clone the repository and organize the datasets as shown in `./data/README.md`. Training, testing, and prediction can all be run from the project root using the `Makefile`.

To train the model, run:

```
make
# (equivalent to make train)
```

To predict 3D bounding boxes on the training/validation/test datasets, run:

```
make predict_on_ground_truth_partition_{train, val, test}
```

To evaluate 3D bounding box predictions, run:

```
make eval_{train, val, test}_predictions
```

To gain familiarity with the model or data, you can also check the notebooks in `./notebooks/`.

## Examples

TODO

## Coming Soon

* Trained model files.
* Model deployment in a Flask application.
* Docker integration to eliminate dependency pains you may or may not experience.
* More robust Makefile to train, test, and predict from both the 2D and 3D bounding box predictor models.
* Plug-and-play project structure: clone, run `make`, and it just works--no need to download data, install packages, or configure obscure settings.
* Unit tests.

## References

TODO
