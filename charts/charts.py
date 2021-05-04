import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("charts/data_holders_18042021.csv")

data_sorted = data.sort_values(by = "Balance", ascending=False).reset_index(drop = True)
lower_holders = data_sorted.iloc[20:,1].sum()
data_to_use = data_sorted.iloc[0:20,1]
data_to_use[len(data_to_use)] = lower_holders
sum_presale = data_sorted.iloc[3:20,1].sum()
data_to_use_presale = data_sorted.iloc[0:3,1]
data_to_use_presale[3] = sum_presale
data_to_use_presale[4] = lower_holders
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = ['Dead', 'Liquidity', 'Dev', '1', '2', '3', '4','5','6','7','8','9',"smaller"]

fig1, ax1 = plt.subplots()
ax1.pie(data_to_use, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

labels_presale = ["Dead", "Liquidity", "Dev", "Top 20 Presale/Whales", "Small Holders"]

fig1, ax1 = plt.subplots()
ax1.pie(data_to_use_presale, labels=labels_presale, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Token Distribution as of April 17th")
plt.show()

labels_presale_circ = ["Liquidity", "Dev", "Top 20 Presale/Whales", "Small Holders"]

fig1, ax1 = plt.subplots()
ax1.pie(data_to_use_presale[1:], labels=labels_presale_circ, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Tokens Distribution on Circ. Supply as of April 18th")
plt.show()

