# Copyright (c) 2026 Banu Darius-Matei
# SPDX-License-Identifier: MIT

# ---------------------------------------------------------- #

import sys
import time
import glob
from pathlib import Path
from paraview.simple import *

MAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MAIN_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_VIDEO_DIR = PROJECT_ROOT / "output-video"

# ---------------------------------------------------------- #

if __name__ == "__main__":
    start_time = time.time()
    output_filenames = str(OUTPUT_DIR / "particles-*.vtk")
    output_video_filename = str(OUTPUT_VIDEO_DIR / "video.mp4")
    files = sorted(glob.glob(output_filenames))
    
    reader = OpenDataFile(files)
    animationScene = GetAnimationScene()
    animationScene.UpdateAnimationUsingDataTimeSteps()
    
    view = GetRenderView()
    view.ViewSize = [1920, 1080]
    view.Background = [0.0, 1.0, 1.0]
    
    display = Show(reader, view)
    display.Representation = 'Point Gaussian'
    display.GaussianRadius = 200
    display.ShaderPreset = 'Plain circle'
    
    ColorBy(display, ('POINTS', 'velocity', 'Magnitude'))
    momentumLUT = GetColorTransferFunction('velocity')
    momentumLUT.ApplyPreset('Cool to Warm', False)
    momentumLUT.RescaleTransferFunction(0.0, 50.0)
    momentumLUT.EnableOpacityMapping = 1
    
    momentumPWF = GetOpacityTransferFunction('velocity')
    momentumPWF.Points = [
        0.0,  0.0, 0.5, 0.0,
        10.0, 1.0, 0.5, 0.0,
        30.0, 1.0, 0.5, 0.0
    ]
    
    view.ResetCamera()
    camera = GetActiveCamera()
    camera.Elevation(60)
    camera.Azimuth(45)
    camera.Zoom(1.5)
    
    print("Rendering animation.")
    SaveAnimation(output_video_filename, view, FrameRate=10, ImageResolution=[1920, 1080])
    total_time = time.time() - start_time
    print("Rendered animation.")
    print(f"Total time taken: {total_time:0.3f}s.\a")

# ---------------------------------------------------------- #