import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def main(xlsx_path="LNP.xlsx"):
    df = pd.read_excel(xlsx_path, header=None, names=["I","H","C","P","Fluorescence"])
    df["Log_Fluorescence"] = np.log1p(df["Fluorescence"])
    df["IP"] = df["I"].astype(str) + "-" + df["P"].astype(str)
    df["HC"] = df["H"].astype(str) + "-" + df["C"].astype(str)

    heatmap_data = df.pivot_table(index="HC", columns="IP", values="Log_Fluorescence", aggfunc="mean")

    plt.figure(figsize=(20,16))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Heatmap of Log-Transformed Fluorescence based on IP and HC")
    plt.xlabel("IP")
    plt.ylabel("HC")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

if __name__ == "__main__":
    main()
