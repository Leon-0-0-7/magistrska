'''
Skripta za prikaz signalov + anonimizacijo signalov + štešje anonimcount + prikaz.
'''

from my_package.lib import (
    generate_signals_with_scipy_skewnorm,
    statistika,
    create_interactive_plot,
    anonymize_signals,
    combine_plots,
    anonim_count,
    plot_signals_with_seaborn
)
import pandas as pd


# GENERIRANJE SIGNALOV: Primer uporabe
# desired_mean = 3.3559
# desired_std = 0.5654
# alpha = -0.4810

# rezult2 = generate_signals_with_scipy_skewnorm(
#     N=30,
#     alpha=alpha,
#     mean=desired_mean,
#     std_dev=desired_std,
#     time=np.arange(
#         0, 10.00, 0.01
#     ),  # 10 sekundni interval, vzorec na 0.01 sekunde - 1000 vzorcev/10s = 100 vzorcev/s
#signal 40HZ, nekaj minut. 50MB exel - več featerjev.
# )
# #shrani v statistika.csv
# rezult2.to_csv('statistika.csv', index=False)

# FILE_NAME = "data/sig-lp_df-v2.xlsx"  # Update the file path to your Excel file
# originalni_signali = pd.read_excel(FILE_NAME)
# originali_signali_brez_indeksa = originalni_signali.iloc[:,1:43].copy()

FILE_NAME = "data/sig_lp_df-v2.csv"
originalni_signali = pd.read_csv(FILE_NAME)
statistika(originalni_signali)
fig1 = create_interactive_plot(originalni_signali)

# Anonimizacija podatkov
meth_code = "quant"
meth_pars = {"min": 0, "max": 10, "dif": 0.5}
Y_df = anonymize_signals(originalni_signali, meth_code, meth_pars)
fig2 = create_interactive_plot(Y_df)

# Kombinacija grafov
plot_list = [fig1, fig2]
titles = ["Osnoven signal", "Anonimiziran signal"]
combined_fig = combine_plots(plot_list, titles, layout=(len(plot_list), 1), height=1000)
combined_fig.show()

# Štetje po anonimizaciji
print(anonim_count(originalni_signali, Y_df))