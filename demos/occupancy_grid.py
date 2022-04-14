import logging
import rospy
import time
import torch
import toros

from visualization_msgs.msg import Marker


if __name__ == '__main__':
    rospy.init_node("occupancy_grid_publisher")
    toros.logging.setup_logging(logging.DEBUG)

    voxel_size = 0.5
    marker_array_pub = rospy.Publisher("/occ_grid", Marker, queue_size=1)

    while not rospy.is_shutdown():
        logging.debug("Creating random occupancy grid")
        occ_grid = (torch.rand((40, 40, 40)) > 0.5).bool()
        logging.debug(f"... number of filled voxel = {occ_grid.sum()}")

        logging.debug("Converting occupancy grid to marker array")
        points = torch.nonzero(occ_grid) * voxel_size
        msg = toros.messages.to_cube_markers(points, scale=voxel_size, ref_frame_id="world", timestamp=rospy.Time.now())

        logging.debug("Publishing occupancy grid")
        marker_array_pub.publish(msg)

        logging.debug("Falling asleep")
        time.sleep(0.1)
