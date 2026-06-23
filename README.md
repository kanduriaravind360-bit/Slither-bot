# Slither Bot V1

A computer vision based Slither.io bot built with Python, OpenCV, MSS, and PyAutoGUI.

## Features

- Real-time screen capture using MSS
- Food detection using HSV color filtering
- Threat detection using contour analysis
- Basic avoidance behavior
- Mouse control through PyAutoGUI

## How It Works

1. Capture the game screen.
2. Detect food pellets.
3. Detect nearby snake bodies.
4. Move toward food when safe.
5. Move away from nearby threats.

## Technologies Used

- Python
- OpenCV
- NumPy
- MSS
- PyAutoGUI
- Tesseract OCR

## Current Limitations

- No enemy head detection
- No path planning
- No reinforcement learning
- Threat detection can be unreliable for some snake colors and skins

## Future Plans

- Improve snake detection
- Detect enemy heads separately
- Add path planning
- Train a reinforcement learning agent
- Create a full autonomous Slither.io AI

## Version

Current release: **V1.0**
