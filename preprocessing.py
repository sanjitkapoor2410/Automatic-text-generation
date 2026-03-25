import pandas as pd
import re
df = pd.read_csv("all_musk_posts.csv")

a = df[['fullText']]
clean_text = []

for tweet in df["fullText"].astype(str):
    tweet = tweet.lower()

    tweet = re.sub(r'https?://S+|www\.\S+','',tweet)

    tweet = re.sub(r'[@#]\S+','',tweet)

    tweet = re.sub(r'[^a-z\s]','',tweet)

    tweet = " ".join(tweet.split())

    if len(tweet)>0:
        clean_text.append(tweet)

print(len(df))
print(len(clean_text))
print(clean_text[:2])