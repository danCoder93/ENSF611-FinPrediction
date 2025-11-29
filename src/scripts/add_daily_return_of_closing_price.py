import pandas as pd

# Load your CSV file
df = pd.read_csv("../../data/vix.csv")
# Convert DATE column to datetime
df["DATE"] = pd.to_datetime(df["DATE"])

# Sort by date (important for correct returns)
df = df.sort_values("DATE")

# ✅ Create Daily Closing Returns
df["Daily_Closing_Return"] = df["CLOSE"].pct_change()

# ✅ Save to the exact filename you asked for
df.to_csv("../../data/vix_with_daily_closing_return.csv", index=False)

print("File saved as: vix_with_daily_closing_return.csv")
