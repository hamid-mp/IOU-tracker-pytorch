# IOU-tracker-pytorch
* Pytorch implementation of simple IOU-tracker
  * Tracking is done per frame.
# tips to get better result
* IOU-tracker as its name suggests, is based on the intersection over union of the bounding boxes, so we want to compare one bbox of currenct frame with previous frame bboxes. 
* to get accurate tracking, the difference between frames should not be so far. so, try to capture frames in higher fps (frame per second) from your video. 
* it is better to have fps higher than 25, then you can choose higher IOU_threshold 

