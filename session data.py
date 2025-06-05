import json
import matplotlib.pyplot as plt
from datetime import datetime

file_path = "total-bitcoins.json"
with open(file_path, "r") as f:
    data = json.load(f)


timestamps = [entry["x"] for entry in data["total-bitcoins"]]
bitcoin_supply = [entry["y"] for entry in data["total-bitcoins"]]


dates = [datetime.fromtimestamp(ts / 1000) for ts in timestamps]


plt.figure(figsize=(12, 6))
plt.plot(dates, bitcoin_supply, color='orange', linewidth=2)
plt.title("Total Bitcoin Supply Over Time", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Total Bitcoins", fontsize=12)
plt.grid(True)
plt.tight_layout()

plt.show()
