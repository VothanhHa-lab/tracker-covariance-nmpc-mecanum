# header.stamp = thời điểm mà các ước lượng được quy về
std_msgs/Header header
TrackedPerson[] people
float64 processed_until   # s, stamp của lô đo cuối đã xử lý
uint32 late_dropped
