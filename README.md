# 3D Object Detection for Autonomous Vehicles
## A FourthBrain AI Project

Authors:
* Leo Kosta: [Leo's github](https://github.com/kostaleonard)
* Ashish Mahashabde: [Ashish's github](https://github.com/amahashabde)

## Project Description

In order to effectively maneuver, autonomous vehicles rely on the ability to accurately estimate bounding boxes of various objects, including other vehicles. Companies like Lyft have recently experimented with using 3D bounding boxes, which could allow autonomous vehicles to have richer positioning information and make better predictions for maneuvering. By reimplementing recent methods and experimenting with new approaches, we want to improve on the state-of-the-art in bounding volume prediction in the self-driving car domain.

## Approach

TODO

## Run Instructions

Clone the repository and organize the datasets as shown in `./data/README.md`. Training, testing, and prediction can all be run from the project root using the `Makefile`.

To train the model, run:

```
make
# (equivalent to make train)
```

To predict 3D bounding boxes on the training/val/test dataset, run:

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
