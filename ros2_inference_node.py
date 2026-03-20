# scripts/ros2_inference_node.py
"""
ROS2 node that subscribes to IMU messages, collects a sliding window, and publishes (fault,severity).
Run after sourcing ROS2:
  source /opt/ros/<distro>/setup.bash
  python3 scripts/ros2_inference_node.py --model models/cnn_multi.pth
Notes: requires rclpy, sensor_msgs.msg, std_msgs.msg
"""
import argparse, collections, numpy as np, time
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--topic", default="/imu/data")
    p.add_argument("--publish_topic", default="/fault_detection")
    p.add_argument("--publish-severity", action="store_true", help="If set, publish severity alongside fault_id when a severity map is available")
    p.add_argument("--severity-map", type=str, default="", help="Optional JSON file mapping fault_id (or label) -> severity int")
    args = p.parse_args()

    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Imu
        from std_msgs.msg import String
    except Exception:
        print("ROS2 python packages not available. Install rclpy and run in a ROS2 environment.")
        raise

    import torch, h5py
    ck = torch.load(args.model, map_location="cpu")
    from cnn_classifier import PaperCNN
    meta = ck.get("meta", {})
    n_faults = meta.get("n_faults", 16)
    model = PaperCNN(in_channels=1, base_filters=32, num_classes=n_faults)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    # normalization stats
    mean = meta.get("mean", None)
    std = meta.get("std", None)
    if mean is not None and std is not None:
        import numpy as _np
        mean = _np.array(mean, dtype="float32")
        std = _np.array(std, dtype="float32")

    class InferenceNode(Node):
        def __init__(self):
            super().__init__("fault_inference_node")
            self.win = collections.deque(maxlen=args.window)
            self.sub = self.create_subscription(Imu, args.topic, self.cb_imu, 10)
            self.pub = self.create_publisher(String, args.publish_topic, 10)
            self.get_logger().info("Fault inference node started.")
            # prepare severity map if requested
            self.sev_map = {}
            if args.publish_severity and args.severity_map:
                try:
                    import json as _json
                    m = _json.load(open(args.severity_map, 'r'))
                    # normalize keys to ints when possible
                    for k,v in m.items():
                        try:
                            ik = int(k)
                        except Exception:
                            ik = k
                        self.sev_map[ik] = int(v)
                    self.get_logger().info(f"Loaded severity map from {args.severity_map}")
                except Exception:
                    self.get_logger().warning("Failed to load severity map file; severity will not be published.")

        def cb_imu(self, msg: Imu):
            # Build feature vector: acc x,y,z then gyro x,y,z. rpm cols not available in standard IMU.
            arr = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                   msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z, 0.0,0.0,0.0,0.0]
            self.win.append(arr)
            if len(self.win) == args.window:
                X = np.array(self.win).T  # (C, W)
                X = X[None, None, :, :].astype("float32")  # (1,1,C,W)
                # apply normalization if available
                if mean is not None and std is not None:
                    try:
                        X = (X - mean) / (std + 1e-9)
                    except Exception:
                        X = (X - mean.reshape(1,1,mean.shape[-2],1)) / (std.reshape(1,1,std.shape[-2],1) + 1e-9)
                inp = torch.from_numpy(X)
                pf = model(inp)
                fid = int(pf.argmax(dim=1).item())
                out = {"fault_id": fid}
                if args.publish_severity:
                    # try sev_map lookup by int id first, then by label from ckpt meta if present
                    sev = None
                    if fid in getattr(self, 'sev_map', {}):
                        sev = self.sev_map[fid]
                    else:
                        # attempt to use checkpoint meta fault_label_map -> label -> severity (if present)
                        try:
                            # ckpt_meta may not be attached here; fallback to no severity
                            ck_meta = meta
                            fm = ck_meta.get('fault_label_map', {})
                            # reverse map: id->label
                            rev = {v:k for k,v in fm.items()} if fm else {}
                            lbl = rev.get(fid, None)
                            if lbl is not None and 'fault_severity_map' in ck_meta:
                                sev = int(ck_meta['fault_severity_map'].get(lbl))
                        except Exception:
                            sev = None
                    if sev is not None:
                        out['severity'] = int(sev)
                self.pub.publish(String(data=str(out)))
    rclpy.init()
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    main()
