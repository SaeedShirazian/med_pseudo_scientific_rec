import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

# Step 1: Load the dataset
df = pd.read_excel('../data/dataset.xlsx')

# Step 2: Basic Inspection
print("First few rows of the dataset:")
print(df.head())
print("\nDataset information:")
df.info()
print("\nMissing values in each column:")
print(df.isnull().sum())

# Step 3: Explore Label Distribution
print("\nLabel distribution:")
print(df['Label'].value_counts())

# Visualization: Bar plot of label distribution
df['Label'].value_counts().plot(kind='bar')
plt.title('Distribution of Labels')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.show()

# Step 4: Analyze Text Lengths
df['Text_Length'] = df['Text'].apply(len)
print("\nText length statistics:")
print(df['Text_Length'].describe())

# Visualization: Histogram of text lengths
plt.hist(df['Text_Length'], bins=30)
plt.title('Distribution of Text Lengths')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

# Step 5: Text Length by Label
print("\nAverage text length by label:")
print(df.groupby('Label')['Text_Length'].mean())

# Visualization: Bar plot of average text length by label
df.groupby('Label')['Text_Length'].mean().plot(kind='bar')
plt.title('Average Text Length by Label')
plt.xlabel('Labels')
plt.ylabel('Average Text Length')
plt.show()

# Step 6: Simple Word Frequency Analysis
all_text = df['Text'].str.cat(sep=' ')
words = all_text.split()
word_counts = Counter(words)
print("\nTop 10 most common words:")
print(word_counts.most_common(10))

# Visualization: Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()