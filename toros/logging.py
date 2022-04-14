import logging


def setup_logging(log_level: int):
    """ROS disables standard Python logging (https://github.com/ros/ros_comm/issues/1384)"""
    console = logging.StreamHandler()
    console.setLevel(log_level)
    logging.getLogger().setLevel(log_level)
    logging.getLogger().addHandler(console)
    formatter = logging.Formatter('[%(asctime)s.%(msecs)03d %(levelname)s] %(message)s', datefmt='%H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
