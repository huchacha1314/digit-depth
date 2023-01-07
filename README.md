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
- Add an LSTM model for predicting slip from collected video frames.
- Add a baseline ResNet based model for estimating total normal force magnitude.
## Config files
There are a number of configs params to be edited before you can run the scripts. This is the rough execution order:
- python scripts/mm_to_pix.py : record a single image and label the distance. Update mm_to_pixel in config file.
- Update calibration ball diameter.
- Update your DIGIT serial_num
- Update gel_width and gel_height
- Update fps
## Usage
Be careful about python path. It is assumed that you run all the scripts from the package folder(/digit-depth) 

Change **gel height,gel width, mm_to_pix, base_img_path, sensor :serial_num ** values in rgb_to_normal.yaml file in config folder.
- `pip install . `
- `python scripts/record.py` : Press SPACEBAR to start recording.
- `python scripts/label_data.py` : Press LEFTMOUSE to label center and RIGHTMOUSE to label circumference.
- `python scripts/create_image_dataset.py` : Create a dataset of images and save it to a csv file.
- `python scripts/train_mlp.py` : Train an MLP model for RGB to Normal mapping.

color2normal model will be saved to a separate folder "models" in /digit-depth/ with its datetime.

After training finishes, update the following config params inside /config/rgb_to_normal.yaml:
- model_path: absolute path to your trained model
- base_img_path: absolute path to a single RGB image without any contact

## Visualization
- python scripts/point_cloud.py : Opens up Open3D screen to visualize point clouds generated by depth image
- python scripts/depth.py : Publishes a ROS topic with the depth image. Modify the params inside for better visualization(threshold values,etc).

 You can also try these ROS nodes to publish RGB image and maximum deformation value from depth images inside /scripts/ros folder:
 ```bash
 - cd scripts/ros
 - python scripts/ros/depth_value_pub.py
 - python scripts/ros/digit_image_pub.py
```
depth_value_pub.py publishes the maximum depth (deformation) value for the entire image when object is pressed. Accuracy depends on your MLP-depth model.
### Please star this repo if you like it!
### Feel free to post an issue and create PRs.
