import pandas as pd
from scipy.stats import skew
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.stats import skewnorm
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots


def statistika(df):
    """
    Izračuna in izpiše globalne statistike za podane signale v DataFrame-u: povprečje, standardni odklon in skewness.
    """
    # 1. Izločite časovni stolpec
    numeric_df = df.drop(columns=["time_s"])

    # 2. Izračunajte globalne statistike
    povprečje = numeric_df.mean().mean()
    standardni_odklon = numeric_df.stack().std()
    skewness = skew(numeric_df.stack())

    print(f"Povprečje vseh meritev: {povprečje:.4f}")
    print(f"Standardni odklon: {standardni_odklon:.4f}")
    print(f"Skewness: {skewness:.4f}\n")


def plot_signals_with_seaborn(df, title="Signal Plot with Seaborn"):
    """
    Funkcija za prikaz signalov iz DataFrame-a z uporabo Seaborn.

    Args:
        df (pandas.DataFrame): DataFrame, ki vsebuje časovni stolpec ('time') in signale.

    Primer uporabe:
    plot_signals_with_seaborn(generiraniSignali)
    """
    # Preoblikuj DataFrame v dolgo obliko za Seaborn
    df_melted = df.melt(id_vars="time_s", var_name="Signal", value_name="Value")

    # Ustvari graf z uporabo Seaborn
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_melted, x="time_s", y="Value", hue="Signal", palette="tab10")

    # Nastavitve grafa
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Signal Value")
    plt.grid(True)
    plt.legend(title="Signals")
    plt.show()


def _adjust_skewnorm_params(desired_mean, desired_std, alpha):
    delta = alpha / np.sqrt(1 + alpha**2)
    adj_scale = desired_std / np.sqrt(1 - 2 * delta**2 / np.pi)
    adj_loc = desired_mean - adj_scale * delta * np.sqrt(2 / np.pi)
    return adj_loc, adj_scale


def _find_alpha(
    target_skew,
):  # alpha v skewnorm ni enak statistični poševnosti (SKEW), temveč parameter, ki posredno vpliva na njo.
    def eq(alpha):
        delta = alpha / np.sqrt(1 + alpha**2)
        skew = (
            (4 - np.pi)
            / 2
            * (delta * np.sqrt(2 / np.pi)) ** 3
            / (1 - 2 * delta**2 / np.pi) ** (3 / 2)
        )
        return skew - target_skew

    result = root_scalar(eq, bracket=[-10, 10])
    return result.root


def generate_signals_with_scipy_skewnorm(
    N, alpha, mean, std_dev, time=np.arange(0, 20.00, 0.02)
):
    """
    Generira pandas DataFrame z N signalov poševne normalne porazdelitve s časovno osjo.

    Parametri:
    ----------
    N : int
        Število generiranih signalov
    alpha : float
        Parameter poševnosti porazdelitve:
        - alpha > 0: pozitivna poševnost (desna porazdelitev)
        - alpha = 0: normalna porazdelitev
        - alpha < 0: negativna poševnost (leva porazdelitev)
    mean : float
        Želeno povprečje generiranih signalov
    std_dev : float
        Želeni standardni odklon signalov
    time : numpy.ndarray, optional
        Časovna os za signal (privzeto: 0-20 sekund s korakom 0.02s)

    Vrača: pandas.DataFrame (time in signali)

    Primer uporabe:
    --------------
    >>> signals = generate_signals_with_scipy_skewnorm(
    ...     N=3,
    ...     alpha=-0.481,
    ...     mean=3.3559,
    ...     std_dev=1.5654,
    ...     time=np.arange(0, 10, 0.01)
    ... )
    >>> print(signals.head())
    """
    # Ustvarimo časovno os
    size = len(time)

    # Inicializiramo DataFrame s časovnim stolpcem
    df = pd.DataFrame({"time_s": time})

    adj_alpha = _find_alpha(alpha)
    adj_loc, adj_scale = _adjust_skewnorm_params(mean, std_dev, adj_alpha)

    # Generiranje posameznih signalov
    for i in range(1, N + 1):
        signal_data = skewnorm.rvs(a=adj_alpha, loc=adj_loc, scale=adj_scale, size=size)
        df[f"signal_{i}"] = signal_data

    return df


# # Primer uporabe
# desired_mean = 3.3559
# desired_std = 0.5654
# alpha = -0.4810

# rezult2 = generate_signals_with_scipy_skewnorm(
#     N=30,
#     alpha=alpha,
#     mean=desired_mean,
#     std_dev=desired_std,
#     time=np.arange(0, 20.00, 0.02),
# )
# statistika(rezult2)
# print(rezult2.head())

# # plot_signals_with_seaborn(rezult2, title="Generirani signali")


def create_interactive_plot(data):
    """
    Create an interactive plot with dropdown for selecting signals.
    data: pandas.DataFrame with time and signal columns.

    # Usage example:
    fig = create_interactive_plot(rezult2)
    """
    # Melt the DataFrame to convert it to long format
    # Ta preoblikovanje je koristno za analizo in vizualizacijo podatkov,
    # saj spremeni podatke iz širokega formata (kjer je vsak signal v svojem stolpcu) v dolgi format (kjer so vsi signali v enem stolpcu)
    melted_df = pd.melt(data, id_vars=["time_s"], var_name="signal", value_name="value")

    # Create the interactive plot with dropdown
    fig = px.line(
        melted_df,
        x="time_s",
        y="value",
        color="signal",
        title="Interactive Signal Selector",
        labels={"time": "Time [s]", "value": "Velikost zenice [mm]"},
        render_mode="webgl"  # Use WebGL rendering

    )

    return fig


def combine_plots(plot_list, titles=None, layout=(1, 1), height=600, width=None, sitetitle="Combined Interactive Plots"):
    """
    Combine multiple plots into a single figure.

    Args:
    plot_list (list): List of dataframes or figures to plot
    titles (list): List of titles for each subplot
    layout (tuple): Number of rows and columns for the subplot layout
    height (int): Height of the combined figure
    width (int): Width of the combined figure (optional)

    Returns:
    plotly.graph_objs._figure.Figure: Combined figure with all plots

    Example:
    fig1 = create_interactive_plot(rezult2)
    fig2 = create_interactive_plot(Y_df)
    # plot_list = [fig1, fig2, another_df, yet_another_df]
    titles = ["Rezult2", "Y_df", "Another Plot", "Yet Another Plot"]
    combined_fig = combine_plots(plot_list, titles, layout=(2, 2), height=800)
    combined_fig.show()
    """
    rows, cols = layout
    subplot_titles = (
        titles if titles else [f"Plot {i+1}" for i in range(len(plot_list))]
    )

    combined_fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    for i, plot_item in enumerate(plot_list):
        row = i // cols + 1
        col = i % cols + 1

        if isinstance(plot_item, pd.DataFrame):
            fig = create_interactive_plot(plot_item)
        else:
            fig = plot_item

        for trace in fig.data:
            combined_fig.add_trace(trace, row=row, col=col)

    combined_fig.update_layout(
        height=height,
        width=width,
        title_text=sitetitle,
    )

    return combined_fig


def anonymize_signals(originalni_signali_df, method_code, method_parameters):
    """
    Funkcija za anonimizacijo signalov.

    Args:
        originalni_signali_df (pandas.DataFrame): DataFrame z originalnimi signali (vključuje stolpec 'time').
        method_code (str): Koda metode za anonimizacijo ('quant' za kvantizacijo).
        method_parameters (dict): Parametri metode {'min': 0, 'max': 10, 'dif': 0.2}.

    Returns:
        anonimizirani_signali_df (pandas.DataFrame): DataFrame z anonimiziranimi signali.

    Primer uporabe:
        # Anonimizacija podatkov
        meth_code = 'quant'
        meth_pars = {'min': 0, 'max': 10, 'dif': 0.5}
        Y_df = anonymize_signals(signals_with_corr, meth_code, meth_pars)
    """
    anonimizirani_signali_df = originalni_signali_df.copy()

    # Podvzorčenje
    subsample_rate = method_parameters.get("subsample_rate", 1)
    anonimizirani_signali_df = anonimizirani_signali_df.iloc[
        ::subsample_rate
    ].reset_index(drop=True)

    if method_code == "quant":
        min_val = method_parameters.get("min", 0)
        max_val = method_parameters.get("max", 10)
        dif = method_parameters.get("dif", 0.2)

        anonimizirani_signali_df.iloc[:, 1:] = (
            np.round((anonimizirani_signali_df.iloc[:, 1:] - min_val) / dif) * dif
            + min_val
        )
        # kvantizacija signal knjižnica?

        # Omeji vrednosti na [min, max]
        anonimizirani_signali_df.iloc[:, 1:] = np.clip(
            anonimizirani_signali_df.iloc[:, 1:], min_val, max_val
        )

    elif method_code == "white_noise":
        std = method_parameters.get("std", 0.5)
        for column in anonimizirani_signali_df.columns[1:]:
            anonimizirani_signali_df[column] = anonimizirani_signali_df[
                column
            ] + np.random.normal(0, std, len(anonimizirani_signali_df))

    return anonimizirani_signali_df


def anonim_count(X_df, Y_df):
    """
    Funkcija za izračun števila anonimiziranih vrednosti.
    TODO:
    Preveri kako knjižnica naredi - npr k-anonimity, l-diversity, t-closeness?

    Args:
        X_df (pandas.DataFrame): DataFrame z originalnimi signali.
        Y_df (pandas.DataFrame): DataFrame z anonimiziranimi signali.

    Returns:
        pandas.DataFrame: Tabela z signal_ID in številom signalov, ki gredo skozi enake razrede anonimizacije.
    """
    meth_code = "quant"
    meth_pars = {"min": 0, "max": 10, "dif": 0.5}
    anonimizirani_original = anonymize_signals(X_df, meth_code, meth_pars)
    results = []

    for column in anonimizirani_original.columns[1:]:
        count = 0
        for column2 in Y_df.columns[1:]:
            if anonimizirani_original[column].equals(Y_df[column2]):
                count += 1
        results.append({"signal_ID": column, "count": count})

    result_df = pd.DataFrame(results)
    return result_df


def filter_signals_based_on_metadata(original_df, metadata_df):
    """
    Filtrira signale v original_df na podlagi časovnih intervalov iz metadata_df.
    
    Args:
        original_df (pd.DataFrame): DataFrame s časom in signali (stolpci: time_s, 38222, 78222...)
        metadata_df (pd.DataFrame): DataFrame s časovnimi intervali (stolpci: Unnamed: 0, t1, t2, t3, t4)
    
    Returns:
        pd.DataFrame: Filtrirani DataFrame z originalnimi časovnimi vrednostmi in samo podatki znotraj intervalov
    """
    result = original_df.copy()
    
    for signal_id in original_df.columns[1:]:  # Preskočimo stolpec 'time_s'
        # Pridobimo čase iz metapodatkov
        try:
            metadata_times = metadata_df.loc[metadata_df['Unnamed: 0'] == int(signal_id), 
                                           ['t1', 't2', 't3', 't4']].values.flatten()
        except:
            print(f"Ne najdem metapodatkov za ID {signal_id}")
            continue
        
        # Filtriranje podatkov
        mask = (
            (original_df['time_s'].between(metadata_times[0], metadata_times[1])) | 
            (original_df['time_s'].between(metadata_times[2], metadata_times[3]))
        )
        
        # Posodobimo rezultat samo za filtrirane vrstice
        result.loc[~mask, signal_id] = np.nan
    
    return result

