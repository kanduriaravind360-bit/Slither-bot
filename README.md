# Slither Bot V2

A computer vision and reinforcement learning based Slither.io bot built with Python, OpenCV, DxCam, PPO, and PyAutoGUI.

## Features

- Real-time screen capture using DxCam
- Food detection using HSV  and gray color filtering
- Threat detection using contour analysis
- Trained through reinforcement learning
- Better avoidance behavior
- Mouse control through PyAutoGUI

## How It Works

1. Capture the observations
2. Detect food pellets.
3. Detect nearby snake bodies.
4. Move toward food when safe.
5. Move away from nearby threats.

## Technologies Used

- OpenCV
- DxCam
- PyAutoGUI
- PPO

## Current Limitations

- Doesn't detect walls that well.
- Sometimes doesn't detect red snakes. 
- Doesn't know how to boost. 
- Doesn't go towards dead snake food pellets.


## Version

Current release: **V2.0**
