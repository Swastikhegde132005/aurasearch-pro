import pandas as pd
from datasets import load_dataset

print("⏳ Downloading 10,000 real AI research papers from Hugging Face...")
# Load a curated dataset of AI papers from Hugging Face
dataset = load_dataset("jamescalam/ai-arxiv", split="train")

# Convert it to a pandas DataFrame
df = pd.DataFrame(dataset)

# Keep the most useful columns and limit to 2,000 papers for local machine speed
df = df[['id', 'title', 'summary']].rename(columns={'summary': 'abstract'})
df = df.head(2000)

# Save it to your folder
df.to_csv("arxiv_sample.csv", index=False)
print("✅ Saved 2,000 real research papers to 'arxiv_sample.csv'!")