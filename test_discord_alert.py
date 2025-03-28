import requests
import matplotlib.pyplot as plt

# Step 1: Generate a sample chart
plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3, 4], [10, 15, 7, 20], marker='o')
plt.title("Sample PnL Chart")
plt.xlabel("Trade #")
plt.ylabel("Profit ($)")
plt.grid(True)
plt.tight_layout()
plt.savefig("sample_pnl_chart.png")
plt.close()

# Step 2: Send to Discord
webhook_url = "https://discord.com/api/webhooks/1353245464866066442/ZmarfW4Tm2wgAuzgmrJV8MR-GdqcZzrXNCtnHCEYvf0ePmn3ZHSVp5uJEVbrdje6C3uh"
with open("sample_pnl_chart.png", "rb") as file:
    response = requests.post(
        webhook_url,
        data={"content": "📈 Test PnL Alert: Here's a sample chart!"},
        files={"file": file}
    )

print("Status Code:", response.status_code)
print("Response:", response.text)
