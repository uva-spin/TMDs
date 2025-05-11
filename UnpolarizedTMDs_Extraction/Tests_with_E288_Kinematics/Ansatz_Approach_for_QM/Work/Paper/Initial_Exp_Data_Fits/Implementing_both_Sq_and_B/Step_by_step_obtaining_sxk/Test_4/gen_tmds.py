import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import lhapdf

# Load LHAPDF Set
NNPDF4_nlo = lhapdf.mkPDF('NNPDF40_nlo_as_01180')

def pdf(pdfset, flavor, x, QQ):
    return pdfset.xfxQ(flavor, x, QQ)

# Constants
QM = 4  # GeV
CSV_FOLDER = "csvs"

# Load n1n2 data
n1n2_file = os.path.join(CSV_FOLDER, "n1n2_grid.csv")
df = pd.read_csv(n1n2_file)

# Ensure x2 = x1
df["x2"] = df["x1"]

df["k"] = df["k"] 

# Compute PDFs
df["f_u_x1"] = df["x1"].apply(lambda x: pdf(NNPDF4_nlo, 2, x, QM))
df["f_ubar_x2"] = df["x2"].apply(lambda x: pdf(NNPDF4_nlo, -2, x, QM))

# Compute fn1 and fn2
df["fn1"] = df["f_u_x1"] * df["n1"]
df["fn2"] = df["f_ubar_x2"] * df["n2"]

# Select and reorder relevant columns
result_df = df[["x1", "x2", "k", "f_u_x1", "f_ubar_x2", "n1", "n2", "fn1", "fn2"]]

# Save result to CSV
output_file = os.path.join(CSV_FOLDER, "TMDS_QM4.csv")
result_df.to_csv(output_file, index=False)

print(f"CSV saved to: {output_file}")







