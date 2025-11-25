"""
Trajectory Collection Script for TMRL Guided Learning

This script records your position as you manually drive around the track.
The trajectory data is saved as a pickle file containing (x, y, z) positions.

Usage:
    1. Start TrackMania and load your track
    2. Run: python collect_trajectory.py
    3. Press ESC to start recording
    4. Drive around the track manually (use full control)
    5. Press ESC again to stop recording and save

The output file can be uploaded to Google Drive and used for reward shaping.
"""

import pickle
import time
import numpy as np
import tmrl
import keyboard
from pathlib import Path

def collect_trajectory(output_file="trajectory_data.pkl"):
    """
    Collect trajectory by recording positions during manual driving

    Args:
        output_file: Path to save the trajectory pickle file
    """
    print("="*60)
    print("TMRL Trajectory Collection")
    print("="*60)
    print("\nInitializing TMRL environment...")

    # Get TMRL environment
    env = tmrl.get_environment()

    print("✓ Environment initialized")
    print("\nInstructions:")
    print("  1. Make sure TrackMania is running with your track loaded")
    print("  2. Press ESC to start recording")
    print("  3. Drive around the track manually")
    print("  4. Press ESC again to stop and save")
    print("\nWaiting for ESC to start recording...")

    # Wait for ESC to start
    keyboard.wait('esc')
    time.sleep(0.5)

    print("\n" + "="*60)
    print("🔴 RECORDING - Drive the track now!")
    print("="*60)
    print("Press ESC to stop recording\n")

    # Reset environment
    env.reset()

    # Trajectory storage
    positions = []
    step_count = 0

    # Get the interface to access raw data
    interface = env.unwrapped.interface

    recording = True
    start_time = time.time()

    # Recording loop
    while recording:
        # Check for ESC to stop
        if keyboard.is_pressed('esc'):
            recording = False
            break

        # Grab current data from game
        try:
            data = interface.client.retrieve_data(sleep_if_empty=0.01, timeout=1.0)

            # Extract position: data[2]=x, data[3]=y, data[4]=z
            position = np.array([data[2], data[3], data[4]], dtype=np.float32)
            positions.append(position)

            step_count += 1

            # Print progress every 100 steps
            if step_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Recorded {step_count} positions ({elapsed:.1f}s)")

        except AttributeError as e:
            if "NoneType" in str(e):
                print(f"\n  Error: TrackMania not detected or OpenPlanet not running")
                print(f"  Please ensure:")
                print(f"    1. TrackMania 2020 is running")
                print(f"    2. OpenPlanet is installed and active")
                print(f"    3. You're on a track (not in menu)")
                break
            else:
                print(f"  Warning: Failed to retrieve data: {e}")
                time.sleep(0.1)
                continue
        except Exception as e:
            print(f"  Warning: Failed to retrieve data: {e}")
            time.sleep(0.1)
            continue

        # Small delay to avoid overwhelming the system
        time.sleep(0.02)  # ~50Hz sampling rate

    # Stop recording
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("⏹ Recording stopped")
    print("="*60)

    # Convert to numpy array
    trajectory = np.array(positions)

    print(f"\nTrajectory Statistics:")
    print(f"  Total positions: {len(trajectory)}")
    print(f"  Recording time: {elapsed:.1f}s")
    print(f"  Sampling rate: {len(trajectory)/elapsed:.1f} Hz")
    print(f"  Trajectory shape: {trajectory.shape}")

    # Save trajectory
    output_path = Path(output_file)
    with open(output_path, 'wb') as f:
        pickle.dump(trajectory, f)

    print(f"\n✓ Trajectory saved to: {output_path.absolute()}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Cleanup
    env.unwrapped.wait()

    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Upload this file to your Google Drive")
    print("2. In remote_trainer.ipynb, set:")
    print(f"   TRAJECTORY_PATH = '/content/drive/MyDrive/.../trajectory_data.pkl'")
    print("3. The trainer will automatically use this for reward shaping")
    print("="*60)

if __name__ == "__main__":
    try:
        collect_trajectory()
    except KeyboardInterrupt:
        print("\n\n✗ Collection interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Collection failed with error: {e}")
        import traceback
        traceback.print_exc()
