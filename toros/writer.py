import logging
import numpy as np
import os
import PIL.Image
import torchvision
import torchvision.transforms.functional as torchvision_transforms
import toros

import rosbag
import rospy

from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from tqdm import tqdm
from typing import Dict, List


class WritableBag(rosbag.Bag):

    def add_images_messages(self, dir_path: str, topic: str, frame_id: str, encoding: str = "rgb8",
                            extension: str = ".png") -> List[rospy.Time]:
        logging.debug(f"Reading image from {dir_path} with encoding {encoding}")
        image_files = os.listdir(dir_path)
        image_files = [f for f in image_files if f.endswith(extension)]
        if len(image_files) == 0:
            return []

        logging.debug(f"... adding {len(image_files)} to bag")
        timestamps = []
        for f in tqdm(image_files):
            f_wo_extension = f.replace(extension, "")
            timestamp = rospy.Time(nsecs=int(f_wo_extension))
            path = os.path.join(dir_path, f)

            if encoding == "rgb8":
                img = torchvision.io.read_image(path)
            elif encoding == "mono8":
                img = PIL.Image.open(path)
                img = torchvision_transforms.to_tensor(img)
            elif encoding == "32FC1":
                img = PIL.Image.open(path)
                img = torchvision_transforms.to_tensor(img) / 1000  # TUM-format with 1000, not 5000 / m
            else:
                raise NotImplementedError(f"Reading mode for encoding {encoding} is not known")
            img_msg = toros.messages.to_image(img, timestamp=timestamp, encoding=encoding, frame_id=frame_id)

            self.write(topic, msg=img_msg, t=timestamp)
            timestamps.append(timestamp)
        return timestamps

    def add_tf_messages(self, file_path: str, frame_id: str, ref_frame_id: str = "world", delimiter: str = ","
                        ) -> List[rospy.Time]:
        logging.debug(f"Reading tf messages from file {file_path} between {frame_id} and {ref_frame_id}")
        poses = np.genfromtxt(file_path, delimiter=delimiter, skip_header=True)
        assert len(poses.shape) == 2
        assert poses.shape[-1] >= 8   # (t, tx, ty, tz, qx, qy, qz, qw)

        logging.debug(f"... loaded poses with shape {poses.shape}")
        for pose_k in poses:
            msg_k = TransformStamped()
            msg_k.header.stamp = rospy.Time(secs=pose_k[0])
            msg_k.header.frame_id = ref_frame_id
            msg_k.child_frame_id = frame_id
            msg_k.transform.translation.x = pose_k[1]
            msg_k.transform.translation.y = pose_k[2]
            msg_k.transform.translation.z = pose_k[3]
            msg_k.transform.rotation.x = pose_k[4]
            msg_k.transform.rotation.y = pose_k[5]
            msg_k.transform.rotation.z = pose_k[6]
            msg_k.transform.rotation.w = pose_k[7]

            tf_msg = TFMessage()
            tf_msg.transforms = [msg_k]
            self.write("/tf", msg=tf_msg, t=msg_k.header.stamp)
        return list(poses[:, 0])

    def add_camera_config_messages(self, camera_dict: Dict, topic: str, frame_id: str, timestamps: List[rospy.Time]
                                   ) -> List[rospy.Time]:
        msg = CameraInfo()
        msg.header.frame_id = frame_id
        msg.width = camera_dict["width"]
        msg.height = camera_dict["height"]
        msg.K = camera_dict["K"]
        msg.D = camera_dict.get("D", [])
        msg.binning_x = camera_dict.get("binning_x", 0)
        msg.binning_y = camera_dict.get("binning_y", 0)

        for t in timestamps:
            msg.header.stamp = t
            self.write(topic, msg=msg, t=t)
        return timestamps
