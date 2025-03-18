import pandas as pd
from scipy.stats import skew

def statistika(df):
    # 1. Izločite časovni stolpec
    numeric_df = df.drop(columns=['time'])

    # 2. Izračunajte globalne statistike
    povprečje = numeric_df.mean().mean()
    standardni_odklon = numeric_df.stack().std()
    poševnost = skew(numeric_df.stack())

    print(f"Povprečje vseh meritev: {povprečje:.4f}")
    print(f"Standardni odklon: {standardni_odklon:.4f}")
    print(f"Poševnost: {poševnost:.4f}\n")


import seaborn as sns
import matplotlib.pyplot as plt

def plot_signals(df, title = "Signal Plot with Seaborn"):
    """
    Funkcija za prikaz signalov iz DataFrame-a z uporabo Seaborn.

    Args:
        df (pandas.DataFrame): DataFrame, ki vsebuje časovni stolpec ('time') in signale.
    """
    # Preoblikuj DataFrame v dolgo obliko za Seaborn
    df_melted = df.melt(id_vars="time", var_name="Signal", value_name="Value")
    
    # Ustvari graf z uporabo Seaborn
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_melted, x="time", y="Value", hue="Signal", palette="tab10")
    
    # Nastavitve grafa
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    plt.grid(True)
    plt.legend(title='Signals')
    plt.show()

# Klic funkcije za prikaz signalov
#plot_signals(generiraniSignali)