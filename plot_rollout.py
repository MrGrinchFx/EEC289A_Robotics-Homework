#!/usr/bin/env python3
"""Custom Evaluation: Plot commanded vs measured velocities from a rollout."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-npz", type=Path, required=True, help="Path to rollout.npz")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/plots"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.rollout_npz)
    
    episodes = np.unique(data["episode_id"])
    episode_labels = ["Forward", "Lateral", "Yaw", "Combined", "Magnitude Sweep"]

    for eid in episodes:
        mask = data["episode_id"] == eid
        
        cmd_vx = data["command_lin_vel_xy"][mask][:, 0]
        meas_vx = data["measured_lin_vel_xy"][mask][:, 0]
        
        cmd_vy = data["command_lin_vel_xy"][mask][:, 1]
        meas_vy = data["measured_lin_vel_xy"][mask][:, 1]
        
        cmd_yaw = data["command_yaw_rate"][mask]
        meas_yaw = data["measured_yaw_rate"][mask]

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Tracking Performance: {episode_labels[eid]} Episode")

        # Plot Forward Velocity (vx)
        axs[0].plot(cmd_vx, label="Commanded vx", linestyle="--", color="black")
        axs[0].plot(meas_vx, label="Measured vx", alpha=0.7, color="blue")
        axs[0].set_ylabel("Velocity (m/s)")
        axs[0].legend()

        # Plot Lateral Velocity (vy)
        axs[1].plot(cmd_vy, label="Commanded vy", linestyle="--", color="black")
        axs[1].plot(meas_vy, label="Measured vy", alpha=0.7, color="green")
        axs[1].set_ylabel("Velocity (m/s)")
        axs[1].legend()

        # Plot Yaw Rate (yaw)
        axs[2].plot(cmd_yaw, label="Commanded yaw", linestyle="--", color="black")
        axs[2].plot(meas_yaw, label="Measured yaw", alpha=0.7, color="red")
        axs[2].set_ylabel("Yaw Rate (rad/s)")
        axs[2].set_xlabel("Time Step")
        axs[2].legend()

        plt.tight_layout()
        plot_path = args.output_dir / f"tracking_plot_ep{eid}.png"
        plt.savefig(plot_path)
        print(f"Saved {plot_path}")

if __name__ == "__main__":
    main()
