"""Pick-and-place client with DWA navigation (lidar obstacle avoidance).

Inherits the full PickPlaceClient pick pipeline but replaces the naive
straight-line route follower with DwaNavigationMixin, which does A* + DWA
lidar avoidance through the obstacle corridor and fast straight following
elsewhere. Use this for the real shelf approach so the robot does not ram the
random boxes.
"""
import argparse
import rclpy

from dwa_navigation import DwaNavigationMixin
from supermarket_sorting_client import PickPlaceClient, NAV_TABLE


class PickPlaceDWAClient(DwaNavigationMixin, PickPlaceClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="kele")
    parser.add_argument("--layout", default=None)
    parser.add_argument("--finish-after-retreat", action="store_true")
    parser.add_argument("--skip-grasp", action="store_true")
    parser.add_argument("--gt-target", action="store_true")
    parser.add_argument("--shelf", default=None)
    parser.add_argument("--prior-fallback", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = PickPlaceDWAClient(kind=args.kind, shelf=args.shelf, layout_path=args.layout,
                             finish_after_retreat=args.finish_after_retreat,
                             skip_grasp=args.skip_grasp, gt_target=args.gt_target,
                             prior_fallback=args.prior_fallback)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
