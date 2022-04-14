import numpy as np
import rospy
import tf2_ros
import tf_conversions
import torch

from geometry_msgs.msg import Point, TransformStamped
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from visualization_msgs.msg import Marker
from typing import Optional, Tuple, Union


def to_pointcloud2(points: torch.Tensor, timestamp: rospy.Time,
                   colors: Optional[Union[torch.Tensor, str]] = None, ref_frame_id: str = "world") -> PointCloud2:
    """Create a point cloud message from a 3D points tensor.

    >>> pc = torch.rand(20, 3)
    >>> t = rospy.Time(10)
    >>> pc_msg = to_pointcloud2(pc, timestamp=t, ref_frame_id="world")
    >>> assert len(pc_msg.fields) == 3
    >>> pc_msg = to_pointcloud2(pc, timestamp=t, ref_frame_id="world", colors="#FF0000")
    >>> assert len(pc_msg.fields) == 6
    >>> pc_msg = to_pointcloud2(pc, colors=torch.zeros(20, 3), timestamp=t)
    >>> assert len(pc_msg.fields) == 6

    Args:
        points: point cloud as torch float tensor [N, 3].
        timestamp: ros message timestamp.
        colors: color of each point in the point cloud [N, 3].
        ref_frame_id: id of point cloud reference frame (in TF tree).
    Returns:
        sensor_msgs::Pointcloud2 message instance.
    """
    assert len(points.shape) == 2
    assert points.shape[-1] == 3
    points_np = points.cpu().detach().numpy().astype(np.float32)
    num_points = len(points_np)

    msg = PointCloud2()
    msg.header.stamp = timestamp
    msg.header.frame_id = ref_frame_id
    msg.height = 1  # unordered
    msg.width = num_points
    msg.is_bigendian = False
    msg.is_dense = True

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]
    msg.point_step = 3 * points_np.dtype.itemsize

    if colors is not None:
        if isinstance(colors, torch.Tensor):
            assert colors.shape == (num_points, 3)
            colors_np = colors.cpu().detach().numpy().astype(np.float32)
        elif isinstance(colors, str):  # hex string
            if not colors.startswith("#"):
                raise ValueError("String color must be hex color string such as #FF0000")
            rgb = tuple(int(colors[1+i:1+i+2], 16) for i in (0, 2, 4))   # offset for skipping "#"
            colors_np = np.ones((num_points, 3), dtype=np.float32)
            colors_np[:, 0] = rgb[0] / 255.0
            colors_np[:, 1] = rgb[1] / 255.0
            colors_np[:, 2] = rgb[2] / 255.0
        else:
            raise NotImplementedError(f"Colors have invalid type {type(colors)}")

        msg.fields += [
            PointField('r', 12, PointField.FLOAT32, 1),
            PointField('g', 16, PointField.FLOAT32, 1),
            PointField('b', 20, PointField.FLOAT32, 1)
        ]
        msg.point_step += 3 * colors_np.dtype.itemsize
        points_np = np.concatenate([points_np, colors_np], axis=-1)

    msg.row_step = msg.point_step * num_points
    msg.data = points_np.tostring()
    return msg


def to_cube_markers(positions: torch.Tensor, timestamp: rospy.Time, scale: Union[Tuple[float, float, float], float],
                    ref_frame_id: str, color: Tuple[float, float, float] = (1.0, 0.0, 0.0), alpha: float = 1.0,
                    namespace: str = "", marker_id: int = 0) -> Marker:
    """Convert position vector to 3D occupancy grid message by using the cube array of the Marker message type.

    Args:
        positions: cube lower left corner position (N, 3).
        timestamp: ros message timestamp.
        scale: size of voxel in x, y, z direction (all voxels have the same size for efficient rendering).
        ref_frame_id: id of point cloud reference frame (in TF tree).
        color: voxel color as RGB color coordinate [0, 1].
        alpha: voxel opacity value [0, 1].
        namespace: marker namespace for identification.
        marker_id: marker id for identification.
    Returns:
        visualization_msgs::Marker message.
    """
    assert len(positions.shape) == 2
    assert positions.shape[-1] == 3
    assert 0 <= alpha <= 1
    assert all(0 <= c <= 1 for c in color)

    if type(scale) == float:
        scale = (scale, scale, scale)

    msg = Marker()
    msg.header.frame_id = ref_frame_id
    msg.header.stamp = timestamp
    msg.type = Marker.CUBE_LIST
    msg.ns = namespace
    msg.id = marker_id

    point_msgs = []
    positions_np = positions.cpu().detach().numpy()
    for point in positions_np:
        point_msg = Point()
        point_msg.x = point[0]
        point_msg.y = point[1]
        point_msg.z = point[2]
        point_msgs.append(point_msg)
    msg.points = point_msgs

    msg.pose.position.x = 0  # center point
    msg.pose.position.y = 0
    msg.pose.position.z = 0
    msg.pose.orientation.x = 0  # center orientation
    msg.pose.orientation.y = 0
    msg.pose.orientation.z = 0
    msg.pose.orientation.w = 1
    msg.scale.x = scale[0]
    msg.scale.y = scale[1]
    msg.scale.z = scale[2]
    msg.color.a = alpha
    msg.color.r = color[0]
    msg.color.g = color[1]
    msg.color.b = color[2]

    return msg


def from_image(msg: Image, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, str, rospy.Time]:
    """Create an image float tensor from an image message.

    >>> img_rgb = torch.randint(0, 255, (3, 4, 2), dtype=torch.uint8)
    >>> img_msg = to_image(img_rgb, timestamp=rospy.Time(10))
    >>> img_rgb2, _, _ = from_image(img_msg)
    >>> assert img_rgb2.shape == img_rgb.shape
    >>> assert img_rgb2.dtype == img_rgb.dtype
    >>> assert torch.allclose(img_rgb, img_rgb2)

    Args:
        msg: image message to convert.
        device: device to ship image to.
    Returns:
        image: RGB tensor with shape (C, H, W).
        frame_id: frame id of image in TF tree.
        timestamp: ros message timestamp.
    """
    if msg.encoding == "rgb8":
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        img = np.transpose(img, (2, 0, 1))
    elif msg.encoding == "32FC1":
        img = np.frombuffer(msg.data, dtype=np.float32).reshape(1, msg.height, msg.width)
    else:
        raise NotImplementedError(f"Conversion not defined for message encoding {msg.encoding}")
    return torch.tensor(img, device=device), msg.header.frame_id, msg.header.stamp


def to_image(img: torch.Tensor, timestamp: rospy.Time, encoding: str = "rgb8", frame_id: str = "camera") -> Image:
    """Create an image message from a float torch image tensor.

    >>> img_rgb = torch.zeros(3, 40, 20)
    >>> t = rospy.Time(10)
    >>> img_msg = to_image(img_rgb, timestamp=t)
    >>> assert img_msg.encoding == "rgb8" and img_msg.height == 40 and img_msg.width == 20
    >>> img_mono = torch.zeros(40, 20)
    >>> img_msg = to_image(img_mono, timestamp=t)
    >>> assert img_msg.encoding == "mono8" and img_msg.height == 40 and img_msg.width == 20

    Args:
        img: image tensor of shape (C, H, W).
        timestamp: ros message timestamp.
        encoding: image encoding.
        frame_id: image TF frame id.
    Returns:
        sensor_msgs::Image instance.
    """
    msg = Image()
    if encoding == "rgb8":
        assert img.shape[0] == 3 and len(img.shape) == 3
        im = img.cpu().detach().numpy().astype(np.uint8)
        msg.step = 3 * im.shape[-1]  # 3 byte channels
    elif encoding == "mono8":
        assert img.shape[0] == 1 or len(img.shape) == 2
        im = img.cpu().detach().numpy().astype(np.uint8)
        msg.step = im.shape[-1]
    elif encoding == "32FC1":
        assert img.shape[0] == 1 or len(img.shape) == 2
        im = img.cpu().detach().numpy().astype(np.float32)
        msg.step = 4 * im.shape[-1]   # 1 float channel = 4 byte channel
    else:
        raise NotImplementedError(f"Conversion not defined for image encoding {encoding}")

    msg.encoding = encoding
    msg.height = im.shape[1]
    msg.width = im.shape[2]
    msg.header.stamp = timestamp
    msg.header.frame_id = frame_id
    msg.is_bigendian = False
    msg.data = np.transpose(im, (1, 2, 0)).tobytes()  # ROS has (H, W, C) format
    return msg


def from_camera_info(msg: CameraInfo, device: Optional[torch.device] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, rospy.Time]:
    """Retrieve the camera parameters (intrinsics, image width and height) from message.

    Args:
        msg: camera info message to convert.
        device: device to ship camera parameters to.
    Returns:
        cam_info: dictionary of camera parameters (intrinsics, image width, image height).
        frame_id: frame id of image in TF tree.
        timestamp: message timestamp as rospy.Time instance.
    """
    if not np.allclose(msg.D, 0):
        raise NotImplementedError("Retrieving distortion parameters is not yet implemented")
    K = torch.tensor(np.array(msg.K).reshape(3, 3), device=device, dtype=torch.float32)
    w = torch.tensor(msg.width, device=device)
    h = torch.tensor(msg.height, device=device)
    return K, h, w, msg.header.frame_id, msg.header.stamp


def broadcast_transform(br: tf2_ros.TransformBroadcaster, T: torch.Tensor, t: rospy.Time,
                        frame_id: str, ref_frame_id: str):
    """Broadcasting transform from `frame_id` to `ref_frame_id`.

    Args:
        br: TF broadcaster.
        T: transform from `ref_frame_id` to `frame_id`.
        t: transformation timestamp as rospy.Time.
        frame_id: child frame id.
        ref_frame_id: reference frame id.
    """
    assert T.shape == (4, 4)
    T_np = T.cpu().detach().numpy()
    trans = T_np[:3, 3]
    quaternion = tf_conversions.transformations.quaternion_from_matrix(T_np)

    msg = TransformStamped()
    msg.header.stamp = t
    msg.header.frame_id = ref_frame_id
    msg.child_frame_id = frame_id
    msg.transform.translation.x = trans[0]
    msg.transform.translation.y = trans[1]
    msg.transform.translation.z = trans[2]
    msg.transform.rotation.x = quaternion[0]
    msg.transform.rotation.y = quaternion[1]
    msg.transform.rotation.z = quaternion[2]
    msg.transform.rotation.w = quaternion[3]
    br.sendTransform(msg)
