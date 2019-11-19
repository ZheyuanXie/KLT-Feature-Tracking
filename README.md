# KLT-Feature-Tracking

## Getting Started
### Dependency
 - OpenCV
 - numpy
 - scipy
 - scikit-image
### Usage

Run `objectTracking.py` to evaluate.

## Method
### Step1 Generate Feature Points
![feature detection](demo/features.png)

### Step2 Estimate Direction of Motion
![optical flow](demo/opticalflow.png)

### Step3 Estimate and Apply Geometric Transform
![affine transformation](demo/affinetransformation.png)

Reject outliers by thresholing.
