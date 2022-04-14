import logging
import numpy as np
import rospy
import tf2_ros
import tf_conversions
import torch

from typing import Optional


class TFListener(tf2_ros.Buffer):

    def __init__(self, **kwargs):
        super(TFListener, self).__init__(**kwargs)
        self.listener = tf2_ros.TransformListener(self)

    def lookup_transform(self, target_frame: str, source_frame: str, time: rospy.Time, timeout=rospy.Duration(0),
                         device: Optional[torch.device] = None) -> torch.Tensor:
        tf_msg = super().lookup_transform(target_frame, source_frame, time=time, timeout=timeout)
        t_W_C = np.array([tf_msg.transform.translation.x,
                          tf_msg.transform.translation.y,
                          tf_msg.transform.translation.z])
        quat_W_C = np.array([tf_msg.transform.rotation.x,
                             tf_msg.transform.rotation.y,
                             tf_msg.transform.rotation.z,
                             tf_msg.transform.rotation.w])

        T_W_C = tf_conversions.transformations.quaternion_matrix(quat_W_C)
        T_W_C[:3, 3] = t_W_C
        return torch.tensor(T_W_C, dtype=torch.float32, device=device)

    def lookup_last_transform(self, target_frame: str, source_frame: str, timeout=rospy.Duration(0),
                              device: Optional[torch.device] = None) -> torch.Tensor:
        t_latest = self.get_latest_common_time(target_frame, source_frame)
        logging.debug(f"Using transformation time {t_latest} for lookup between {target_frame} and {source_frame}")
        return self.lookup_transform(target_frame, source_frame, time=t_latest, timeout=timeout, device=device)

    def frame_exists(self, frame_id: str) -> bool:
        return self._frameExists(frame_id)


class TFContext:

    def __init__(self):
        self.succeeded = True

    def __enter__(self):
        self.succeeded = True
        return self

    def __exit__(self, value_type, value, traceback):
        if value_type in (tf2_ros.LookupException, tf2_ros.TimeoutException, tf2_ros.ExtrapolationException):
            logging.warning(value)
            self.succeeded = False
        return True

