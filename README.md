<br />
<p align="center">
  <img src="https://github.com/vocdex/vocdex.github.io/blob/master/assets/img/icon.png" width="150" title="hover text">
</p>

# DIGIT
This codebase allows you:
- Collect image frames from DIGIT and annotate circles in each frame.
- Save the annotated frame values into a csv file.
- Train a baseline MLP model for RGB to Normal mapping.
- Generate depth maps in real-time using a fast Poisson Solver.
- Estimate 2D object pose using PCA and OpenCV built-in algorithms.

Currently, labeling circles is done manually for each sensor. It can take up to an hour for annotating 30 images.  
This codebase has a script that will replace manual labeling and model training process up to 10 mins.
## Visualization
### Estimating object pose by fitting an ellipse (PCA and OpenCV):
<br />
<p align="center">
  <img src="https://github.com/vocdex/digit-depth/blob/main/assets/depthPCA.gif" width="400" title="depth">
</p>

### Depth image point cloud :
<br />
<p align="center">
  <img src="https://github.com/vocdex/digit-depth/blob/main/assets/point-cloud.gif" width="400" title="point-cloud">
</p>

### Marker movement tracking ( useful for force direction and magnitude estimation):
<br />
<p align="center">
  <img src="https://github.com/vocdex/digit-depth/blob/main/assets/markers.gif" width="400" title="marker">
</p>

## TODO
- Add a Pix2Pix model to generate depth maps from RGB images.
- Add a Monocular Depth model to generate depth maps from RGB images.
## Config files
There are a number of configs params to be edited before you can run the scripts. This is the rough execution order:
- `python scripts/mm_to_pix.py` : This script will help you calculate the mm_to_pix value for your sensor. You need to place a caliper on the sensor and press SPACEBAR to capture the image. Then, you need to enter the distance between the two ends of the caliper in mm. This will give you the mm_to_pix value for your sensor.Replace the value in config/digit.yaml file.
Other config params in digit.yaml:
- `gel_height`: Height of the gel in mm
- `gel_width`: Width of the gel in mm
- `gel_thickness`: Thickness of the gel in mm
- `gel_min_depth`: Minimum depth of the gel in mm (max deformation)
- `ball_diameter`: Diameter of the calibration ball in mm
- `max_depth`: Maximum depth of the gel in mm (min deformation)
- `sensor/serial_num`: Serial number of the sensor
- `sensor/fps`: Frames per second of the sensor. Default is 30. There are some issues with 60 FPS.

## Usage
Be careful about python path. It is assumed that you run all the scripts from the package folder(/digit-depth) 

After changing the config params, run the following scripts in the following order:

- `pip install -r requirements.txt`
- `pip install . `
Now you should have the package installed in your python environment.
To train the model, you need to collect data first. You can use the following scripts to collect data:
- `source /home/jiaixn.hu/anaconda3/bin/activate` .
- `conda activate digit` .
- `cd Tactile/digit-depth/` .
  

- `python scripts/record.py` : Press SPACEBAR to start recording. Collect 30-40 images.
- `python scripts/label_data.py` : Press LEFTMOUSE to label center and RIGHTMOUSE to label circumference.
- `python scripts/create_image_dataset.py` : Create a dataset of images and save it to csv files.
- `python scripts/train_mlp.py` : Train an MLP model for RGB to Normal mapping.

color2normal model will be saved to a separate folder "models" in /digit-depth/ with its datetime.

## Visualization
- `cd ws_moveit/`: 随便打开一个包含ros的工程
- `source devel/setup.bash`
- `roscore`
新开一个terminal
- `cd `
- `cd ws_moveit/`: 随便打开一个包含ros的工程
- `source devel/setup.bash`
- `cd Tactile/digit-depth`:重新打开digit-depth工程
- `source /home/jiaixn.hu/anaconda3/bin/activate`
- `conda activate digit`
  接下来运行
- `python scripts/point_cloud.py `: Opens up Open3D screen to visualize point clouds generated by depth image
- `python scripts/depth.py` : Publishes a ROS topic with the depth image. Modify the params inside for better visualization(threshold values,etc).

 You can also try these ROS nodes to publish RGB image and maximum deformation value from depth images inside /scripts/ros folder:
 
 - `python scripts/ros/depth_value_pub.py`: Publishes the maximum depth (deformation) value for the entire image when object is pressed. Accuracy depends on your MLP-depth model.
 - `python scripts/ros/digit_image_pub.py`: Publishes the RGB image from the sensor.

## Issues
- If you are using a 60 FPS sensor, you might need to change the fps value in config/digit.yaml file. There are some issues with 60 FPS. Refer to this [issue](https://github.com/facebookresearch/digit-interface/issues/10)
- MLP model accuracy really depends on the quality of RGB lighting. If you have produced your own DIGIT, make sure the light is not directly hitting the DIGIT internal camera. 
## Acknowledgements
I have modified/used the code from the following repos:
- [digit-interface](https://github.com/facebookresearch/digit-interface)
- [pytouch](https://github.com/facebookresearch/PyTouch)
- [PoissonReconstruction](https://gist.github.com/jackdoerner/b9b5e62a4c3893c76e4c)
- [tactile-inhand](https://github.com/psodhi/tactile-in-hand)

### Feel free to post an issue and create PRs.
