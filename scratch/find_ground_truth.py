import glob
import os

print("JSON files:")
for f in glob.glob("**/*.json", recursive=True):
    print(f"  {f}")

print("\nCSV files:")
for f in glob.glob("**/*.csv", recursive=True):
    print(f"  {f}")
