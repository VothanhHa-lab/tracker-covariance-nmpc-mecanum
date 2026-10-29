# header.stamp = MEASUREMENT TIME of the sensor (not the processing time)
std_msgs/Header header
string sensor      # "lidar" | "camera"
PersonDetection[] detections
