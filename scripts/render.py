# Copyright (c) 2026 Banu Darius-Matei
# SPDX-License-Identifier: MIT

# ---------------------------------------------------------- #

import sys
import glob
import subprocess
from pathlib import Path
from paraview.simple import *

MAIN_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = MAIN_DIR / "output"
OUTPUT_VIDEO_DIR = MAIN_DIR / "output-video"

# ---------------------------------------------------------- #

def render_paraview():
    output_filenames = str(OUTPUT_DIR / "particles-*.vtk")
    output_video_filename = str(OUTPUT_VIDEO_DIR / "video.mp4")
    frames_filename = str(OUTPUT_VIDEO_DIR / "frame.png")
    ffmpeg_input = str(OUTPUT_VIDEO_DIR / "frame.%04d.png")
    
    files = sorted(glob.glob(output_filenames))
    
    reader = OpenDataFile(files)
    animationScene = GetAnimationScene()
    animationScene.UpdateAnimationUsingDataTimeSteps()
    
    view = GetRenderView()
    view.ViewSize = [1920, 1080]
    LoadPalette('WarmGrayBackground')
    
    display = Show(reader, view)
    display.Representation = 'Point Gaussian'
    display.GaussianRadius = 400
    display.ShaderPreset = 'Plain circle'
    
    ColorBy(display, ('POINTS', 'velocity', 'Magnitude'))
    momentumLUT = GetColorTransferFunction('velocity')
    momentumLUT.ApplyPreset('Cool to Warm (Extended)', False)
    momentumLUT.RescaleTransferFunction(0.0, 50.0)
    momentumLUT.EnableOpacityMapping = 1
    
    momentumPWF = GetOpacityTransferFunction('velocity')
    momentumPWF.Points = [
        0.0,  0.0, 0.5, 0.0,
        15.0,  0.0, 0.5, 0.0,
        20.0, 1.0, 0.5, 0.0,
        30.0, 1.0, 0.5, 0.0
    ]
    
    view.ResetCamera()
    camera = GetActiveCamera()
    camera.Elevation(40)
    camera.Azimuth(45)
    camera.Zoom(2)
    
    print("Rendering animation frames.")
    SaveAnimation(frames_filename, view, FrameRate=10, ImageResolution=[1920, 1080])
    print("Rendered animation frames.")
    
    ffmpeg_command = ["ffmpeg", "-y", "-framerate", "10", "-i", ffmpeg_input, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-loglevel", "error", output_video_filename]
    subprocess.run(ffmpeg_command, text=True)
    
    for frame in OUTPUT_VIDEO_DIR.glob("frame.*.png"):
        frame.unlink()

# ---------------------------------------------------------- #