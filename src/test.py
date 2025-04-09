import pandas as pd
import numpy as np
print("Hello World")

from my_package.lib import statistika

import pandas as pd

FILE_NAME = "data/sig-lp_df-v2.xlsx"  # Update the file path to your Excel file
generirani_signali = pd.read_excel(FILE_NAME)

# To inspect the first few rows of the DataFrame
# print(generirani_signali.head())