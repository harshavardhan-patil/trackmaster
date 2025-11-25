import tmrl

env = tmrl.get_environment()

# Traverse wrapper chain
e = env
depth = 0
while hasattr(e, 'env'):
    print(f"Layer {depth}: {type(e)}")
    e = e.env
    depth += 1

print(f"Layer {depth}: {type(e)}")
print(f"\nAttributes: {[x for x in dir(e) if not x.startswith('_')][:30]}")

# Check for interface
if hasattr(e, 'interface'):
    print(f"\nFound interface: {type(e.interface)}")
    if hasattr(e.interface, 'client'):
        print(f"  Found interface.client: {type(e.interface.client)}")
        print(f"  Trying to retrieve data...")
        try:
            data = e.interface.client.retrieve_data(sleep_if_empty=0.01, timeout=0.5)
            print(f"  Success! Data length: {len(data)}")
            print(f"  Position data [2:5]: {data[2:5]}")
        except Exception as ex:
            print(f"  Error: {ex}")
    else:
        print(f"  No client attribute on interface")
else:
    print(f"\nNo interface attribute found")
