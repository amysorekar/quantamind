import pandas as pd
import os
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/support_data.csv", dtype={
    "tweet_id": str,
    "in_response_to_tweet_id": str
})

df = df.dropna(subset=["text", "in_response_to_tweet_id", "tweet_id"])

df["tweet_id"] = df["tweet_id"].astype(str)
df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].astype(str)

msg_lookup = df.set_index("tweet_id")["text"].to_dict()
inbound_lookup = df.set_index("tweet_id")["inbound"].to_dict()

pairs = []
for _, row in df.iterrows():
    msg_id = row["tweet_id"]
    in_response_to = row["in_response_to_tweet_id"]

    if not in_response_to in msg_lookup:
        continue

    if row["inbound"] == False and inbound_lookup.get(in_response_to) == True:
        customer_msg = msg_lookup[in_response_to]
        brand_reply = row["text"]
        pairs.append({"input": customer_msg, "output": brand_reply})

print(f"Collected {len(pairs)} input-output pairs")


df_clean = pd.DataFrame(pairs).sample(n=500, random_state=42)

train_df, temp_df = train_test_split(df_clean, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

os.makedirs("data", exist_ok=True)
train_df.to_json("data/train.json", orient="records", lines=True)
val_df.to_json("data/val.json", orient="records", lines=True)
test_df.to_json("data/test.json", orient="records", lines=True)

print(f"Saved: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
