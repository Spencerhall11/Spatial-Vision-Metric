# Spatial-Vision-Metric
I designed this for processing images 

Key Features:
Static Image Interface: used Ultralytics YOLO v8
Pixels Per Metric Calibrating: used an algorith to compare items using a reference object
OpenCV pre-processing: used CV2 to sclae and filter and minimize data skew and box issues
Batch Processing: designed to be able to handle multiple inputs

This was used as an exercise to learn more about machine learning and the associated models
this uses an img input containing an item of a known size and the ultralytics database to confirm the object is present.
OpenCV will import the image and read it then analysis is done with numpy and cv2
