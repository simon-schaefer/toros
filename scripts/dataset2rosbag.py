import argparse
import logging
import os
import toros


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset directory.")
    return parser.parse_args()


def main(args):
    f_name = os.path.join(args.dataset, "okvis.bag")
    cam_pose_file = os.path.join(args.dataset, "okvis2_trajectory_rgb.csv")

    rgb_camera_dict = {
        "width": 640,
        "height": 480,
        "K": [379.4777373482001, 0.0, 323.15375716553234, 0.0, 380.14695054162087, 246.10592930037143, 0.0, 0.0, 1.0]
    }

    depth_camera_dict = {
        "width": 640,
        "height": 480,
        "K": [385.070495605469, 0.0, 318.642700195312, 0.0, 385.070495605469, 239.911026000977, 0.0, 0.0, 1.0]
    }

    bag = toros.writer.WritableBag(f_name, mode="w")
    ts = bag.add_images_messages(os.path.join(args.dataset, "depth0", "data"), "/depth_cam/raw", "depth0", encoding="32FC1")
    _ = bag.add_camera_config_messages(depth_camera_dict, topic="/depth_cam/camera_info", frame_id="depth0", timestamps=ts)
    ts = bag.add_images_messages(os.path.join(args.dataset, "rgb0", "data"), "/camera/rgb", "rgb0", encoding="rgb8")
    _ = bag.add_camera_config_messages(rgb_camera_dict, topic="/camera/camera_info", frame_id="rgb0", timestamps=ts)
    _ = bag.add_tf_messages(cam_pose_file, "imu0", ref_frame_id="world")
    bag.close()


if __name__ == '__main__':
    args_ = parse_args()
    toros.logging.setup_logging(logging.INFO)
    main(args_)
