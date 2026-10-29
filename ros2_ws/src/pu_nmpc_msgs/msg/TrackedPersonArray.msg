# header.stamp = time to which the estimates refer
std_msgs/Header header
TrackedPerson[] people
float64 processed_until   # s, stamp of the last processed measurement batch
uint32 late_dropped
