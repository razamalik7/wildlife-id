import os

print(f"Current Working Directory: {os.getcwd()}")

target = "training_data"
if os.path.exists(target):
    print(f"✅ Found '{target}' folder.")
    print(f"Contents of '{target}': {os.listdir(target)}")

    # Check for 'train' specifically
    if os.path.exists(os.path.join(target, "train")):
        print("✅ Found 'train' folder inside.")
    else:
        print("❌ MISSING 'train' folder inside.")
else:
    print(f"❌ Could not find '{target}' folder at all.")

# Check if maybe it's in the old default folder
if os.path.exists("dataset_raw"):
    print("⚠️ Found 'dataset_raw' folder! Your data might be here instead.")