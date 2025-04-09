'''
Skripta za razrez signalov glede na metapodatke + anonimizacijo signalov +  prikaz.
'''
from my_package.lib import (
    generate_signals_with_scipy_skewnorm,
    statistika,
    create_interactive_plot,
    anonymize_signals,
    combine_plots,
    anonim_count,
    plot_signals_with_seaborn,
    filter_signals_based_on_metadata,
)
import pandas as pd

FILE_NAME = "data/sig_lp_df-v2.csv"
originalni_signali = pd.read_csv(FILE_NAME)
kopija_original_signal_id_st = originalni_signali.copy()

# Extract numeric part from column names
kopija_original_signal_id_st.columns = [
    col.split("_")[0] if "_lp_spl" in col else col
    for col in kopija_original_signal_id_st.columns
]

FILE_NAME = "data/md-lp_df.xlsx"  # Update the file path to your Excel file
metadata_podatki = pd.read_excel(FILE_NAME)

filtrirani_df = filter_signals_based_on_metadata(
    kopija_original_signal_id_st, metadata_podatki
)
fig1 = create_interactive_plot(filtrirani_df)
# fig1.show()

# Anonimizacija podatkov


meth_code = "quant"
meth_pars = {"min": 0, "max": 10, "dif": 0.5}
Y_df = anonymize_signals(filtrirani_df, meth_code, meth_pars)
fig2 = create_interactive_plot(Y_df)

# Kombinacija grafov
plot_list = [fig1, fig2]
titles = ["Osnoven signal", "Anonimiziran signal"]
combined_fig = combine_plots(plot_list, titles, layout=(len(plot_list), 1), height=1000, sitetitle="Graf originalnih signalov na relevantnih intervalih in graf anonimiziranih signalov")
combined_fig.show()
