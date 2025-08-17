import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def main(xlsx_path="LNP.xlsx"):
    df = pd.read_excel(xlsx_path, header=None, names=["I","H","C","P","Fluorescence"])
    df["Log_Fluorescence"] = np.log1p(df["Fluorescence"])

    axes_combinations = [
        (("I","H"),("C","P")),
        (("I","C"),("H","P")),
        (("I","P"),("H","C")),
        (("H","C"),("I","P")),
        (("H","P"),("I","C")),
        (("C","P"),("I","H")),
    ]

    def combine_cols(df, c1, c2):
        return df[c1].astype(str) + "-" + df[c2].astype(str)

    plt.figure(figsize=(18,12))
    for i, ((x1,x2),(y1,y2)) in enumerate(axes_combinations, start=1):
        plt.subplot(2,3,i)
        x = combine_cols(df, x1, x2)
        y = combine_cols(df, y1, y2)
        sns.scatterplot(x=x, y=y, hue="Log_Fluorescence", palette="viridis", data=df, s=40, edgecolor="none")
        plt.title(f"{x1}-{x2} vs {y1}-{y2}")
        plt.xlabel(f"{x1}-{x2}")
        plt.ylabel(f"{y1}-{y2}")
        plt.xticks(rotation=90)
        plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
