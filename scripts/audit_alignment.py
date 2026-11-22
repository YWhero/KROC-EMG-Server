import argparse
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    args = p.parse_args()
    result = {}
    for ep in sorted((p for p in args.root.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)):
        entry = {}
        with h5py.File(ep / "emg/segments/0/derived/emg_all_variants_v1.h5", "r") as f:
            mono = f["native/mapped_host_ns"][:]
            recv = f["native/host_receive_ns"][:]
            wall = f["native/host_receive_system_ns"][:]
            # Each receive event contains both clock readings; no floating epoch math.
            acquisition = mono + (wall - recv)
            entry["emg"] = {
                "samples": len(mono), "clock_offset_range_ns": [int((wall-recv).min()), int((wall-recv).max())],
                "acquisition_span_ns": [int(acquisition[0]), int(acquisition[-1])],
                "receive_minus_acquisition_ms": np.percentile((wall-acquisition)/1e6, [0,50,99,100]).tolist(),
                "non_increasing_acquisition": int((np.diff(acquisition) <= 0).sum()),
                "notch_valid_samples": int(f["quality/notch_warmup_valid"][:].sum())}
        cameras = {}
        for path in sorted(ep.glob("videos/*/*_timestamps.parquet")):
            df = pd.read_parquet(path)
            video = path.with_name(path.name.replace("_timestamps.parquet", ".mp4"))
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=width,height,r_frame_rate,nb_frames", "-of", "json", str(video)], text=True))
            cameras[path.stem] = {"frames": len(df), "video": probe,
                                  "span_ns": [int(df.recv_ns.iloc[0]), int(df.recv_ns.iloc[-1])],
                                  "receive_minus_header_ms": np.percentile(
                                      (df.recv_ns.to_numpy()-df.header_stamp_ns.to_numpy())/1e6, [0,50,99,100]).tolist()}
        entry["cameras"] = cameras
        if ep.name == "0":
            with AnyReader([ep], default_typestore=get_typestore(Stores.ROS2_JAZZY)) as reader:
                wanted = [c for c in reader.connections if c.topic in
                          ["/left/joint_states", "/left/leader/joint_trajectory"]]
                counts, examples = {}, {}
                for connection, timestamp, raw in reader.messages(connections=wanted):
                    counts[connection.topic] = counts.get(connection.topic, 0) + 1
                    if connection.topic not in examples:
                        msg = reader.deserialize(raw, connection.msgtype)
                        examples[connection.topic] = {"record_ns": timestamp, "message": str(msg)}
                entry["robot"] = {"counts": counts, "first": examples}
        result[ep.name] = entry
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
