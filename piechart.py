import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('ABC.xlsx - Category - ABC.csv', skiprows=1)  # Skip the first row which is a title

# Clean the data by removing rows with NaN values in Product Category
df = df.dropna(subset=['Product Category'])

# Remove the "Grand Total" row if it exists
df = df[df['Product Category'] != 'Grand Total']

print("Data loaded successfully!")
print(f"Total number of categories: {len(df)}")
print(f"ABC Classifications: {df['ABC Classification'].value_counts()}")

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 1. Pie chart for ABC Classification
abc_counts = df['ABC Classification'].value_counts()
colors_abc = ['#ff9999', '#66b3ff', '#99ff99']  # Light colors for A, B, C

wedges1, texts1, autotexts1 = ax1.pie(abc_counts.values, 
                                      labels=abc_counts.index, 
                                      autopct='%1.1f%%',
                                      colors=colors_abc,
                                      startangle=90,
                                      explode=(0.05, 0.05, 0.05))  # Slightly separate each slice

ax1.set_title('Distribution by ABC Classification', fontsize=14, fontweight='bold', pad=20)

# Enhance the appearance of the first pie chart
for autotext in autotexts1:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

# 2. Pie chart for Product Categories (top categories)
# Since there are many categories, let's show top 10 and group the rest as "Others"
category_counts = df['Product Category'].value_counts()

# Take top 10 categories
top_categories = category_counts.head(10)
other_count = category_counts.tail(len(category_counts) - 10).sum()

# Create data for the second pie chart
if other_count > 0:
    pie_data = list(top_categories.values) + [other_count]
    pie_labels = list(top_categories.index) + ['Other Categories']
else:
    pie_data = list(top_categories.values)
    pie_labels = list(top_categories.index)

# Generate colors for the categories
colors_cat = plt.cm.Set3(np.linspace(0, 1, len(pie_data)))

wedges2, texts2, autotexts2 = ax2.pie(pie_data, 
                                      labels=pie_labels, 
                                      autopct='%1.1f%%',
                                      colors=colors_cat,
                                      startangle=90)

ax2.set_title('Distribution by Product Category (Top 10)', fontsize=14, fontweight='bold', pad=20)

# Enhance the appearance of the second pie chart
for autotext in autotexts2:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(8)

# Adjust text size for category labels (they might be long)
for text in texts2:
    text.set_fontsize(8)

# Add a main title for the entire figure
fig.suptitle('Inventory Analysis: ABC Classification & Product Categories', 
             fontsize=16, fontweight='bold', y=0.95)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

# Print some statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
print(f"\nABC Classification Distribution:")
for classification, count in abc_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {classification}: {count} categories ({percentage:.1f}%)")

print(f"\nTop 5 Product Categories by Count:")
for i, (category, count) in enumerate(category_counts.head(5).items(), 1):
    print(f"  {i}. {category}: {count} item(s)")

# Create a detailed breakdown chart
plt.figure(figsize=(12, 8))

# Create a stacked bar chart showing ABC classification within each top category
top_10_categories = category_counts.head(10).index
abc_breakdown = {}

for abc_class in ['A', 'B', 'C']:
    abc_breakdown[abc_class] = []
    for category in top_10_categories:
        count = len(df[(df['Product Category'] == category) & (df['ABC Classification'] == abc_class)])
        abc_breakdown[abc_class].append(count)

x_pos = np.arange(len(top_10_categories))
width = 0.6

# Create stacked bars
p1 = plt.bar(x_pos, abc_breakdown['A'], width, label='Class A', color='#ff9999')
p2 = plt.bar(x_pos, abc_breakdown['B'], width, bottom=abc_breakdown['A'], label='Class B', color='#66b3ff')
p3 = plt.bar(x_pos, abc_breakdown['C'], width, 
            bottom=[a + b for a, b in zip(abc_breakdown['A'], abc_breakdown['B'])], 
            label='Class C', color='#99ff99')

plt.xlabel('Product Categories', fontweight='bold')
plt.ylabel('Number of Items', fontweight='bold')
plt.title('ABC Classification Breakdown by Top 10 Product Categories', fontweight='bold', pad=20)
plt.xticks(x_pos, top_10_categories, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()