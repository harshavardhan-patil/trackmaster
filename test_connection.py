"""
Test script to verify connection to remote training server

Usage:
    python test_connection.py https://your-ngrok-url.ngrok.io
"""

import sys
import time
import pickle
import requests
import numpy as np

def test_status_endpoint(remote_url):
    """Test the /status endpoint"""
    print("1. Testing /status endpoint...")
    try:
        response = requests.get(f"{remote_url}/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Status: {data['status']}")
            print(f"   ✓ Update count: {data['update_count']}")
            print(f"   ✓ Latest reward: {data['latest_reward']}")
            return True
        else:
            print(f"   ✗ Failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_reset_endpoint(remote_url):
    """Test the /reset endpoint"""
    print("\n2. Testing /reset endpoint...")
    try:
        response = requests.post(f"{remote_url}/reset", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Reset successful: {data['status']}")
            return True
        else:
            print(f"   ✗ Failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_action_endpoint(remote_url):
    """Test the /action endpoint with dummy observation"""
    print("\n3. Testing /action endpoint...")
    try:
        # Create dummy observation (same format as TMRL)
        dummy_obs = (
            np.array([50.0]),  # speed
            np.array([3.0]),   # gear
            np.array([100.0]), # rpm
            np.random.rand(4, 64, 64).astype(np.float32),  # images
            np.array([0.0, 0.0, 0.0]),  # previous action 1
            np.array([0.0, 0.0, 0.0])   # previous action 2
        )

        # Serialize and send
        data = pickle.dumps(dummy_obs)
        response = requests.post(
            f"{remote_url}/action",
            data=data,
            headers={'Content-Type': 'application/octet-stream'},
            timeout=10
        )

        if response.status_code == 200:
            result = pickle.loads(response.content)
            action = result['action']
            logprob = result['logprob']
            state_value = result['state_value']

            print(f"   ✓ Action received: {action}")
            print(f"   ✓ Log probability: {logprob:.4f}")
            print(f"   ✓ State value: {state_value:.4f}")

            # Verify action is in valid range
            if np.all(action >= -1.5) and np.all(action <= 1.5):
                print(f"   ✓ Action values in valid range")
                return True
            else:
                print(f"   ⚠ Warning: Action values outside expected range")
                return True
        else:
            print(f"   ✗ Failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_action_latency(remote_url, num_requests=10):
    """Test action request latency"""
    print(f"\n4. Testing action latency ({num_requests} requests)...")
    try:
        # Create dummy observation
        dummy_obs = (
            np.array([50.0]),
            np.array([3.0]),
            np.array([100.0]),
            np.random.rand(4, 64, 64).astype(np.float32),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0])
        )
        data = pickle.dumps(dummy_obs)

        latencies = []
        for i in range(num_requests):
            start_time = time.time()
            response = requests.post(
                f"{remote_url}/action",
                data=data,
                headers={'Content-Type': 'application/octet-stream'},
                timeout=10
            )
            latency = (time.time() - start_time) * 1000  # Convert to ms

            if response.status_code == 200:
                latencies.append(latency)
            else:
                print(f"   ⚠ Request {i+1} failed")

        if latencies:
            avg_latency = np.mean(latencies)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)

            print(f"   ✓ Average latency: {avg_latency:.1f} ms")
            print(f"   ✓ Min latency: {min_latency:.1f} ms")
            print(f"   ✓ Max latency: {max_latency:.1f} ms")

            if avg_latency < 100:
                print(f"   ✓ Latency is good for real-time training")
            elif avg_latency < 200:
                print(f"   ⚠ Latency is acceptable but may slow training")
            else:
                print(f"   ⚠ High latency - training may be slow")

            return True
        else:
            print(f"   ✗ All requests failed")
            return False

    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_connection.py <remote_url>")
        print("Example: python test_connection.py https://abc123.ngrok.io")
        sys.exit(1)

    remote_url = sys.argv[1].rstrip('/')

    print("="*60)
    print("Testing Connection to Remote Training Server")
    print("="*60)
    print(f"\nRemote URL: {remote_url}\n")
    print("="*60)

    # Run tests
    results = []
    results.append(("Status Endpoint", test_status_endpoint(remote_url)))
    results.append(("Reset Endpoint", test_reset_endpoint(remote_url)))
    results.append(("Action Endpoint", test_action_endpoint(remote_url)))
    results.append(("Action Latency", test_action_latency(remote_url)))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n✓ All tests passed! You can start training.")
        print("\nRun:")
        print(f"  python local_trainer.py --remote-url {remote_url} --num-updates 10000")
    else:
        print("\n✗ Some tests failed. Please check:")
        print("  1. Colab notebook is running (Cell 6)")
        print("  2. ngrok URL is correct")
        print("  3. No firewall blocking the connection")

    print()

if __name__ == "__main__":
    main()
