import pandas as pd
import matplotlib.pyplot as plt

# Physical Constants
PLANCK_CONSTANT = 6.626e-34  # in Joule seconds
SPEED_OF_LIGHT = 3e8         # in meters per second
AVOGADRO_NUMBER = 6.022e23   # number of entities per mole

# Load the data
df = pd.read_excel('Solar_Spectrum.xls', skiprows=1)

# Data Inspection and Cleaning
print(df.head())
print(df.columns)
print(df.dtypes)
df['Wavelength'] = pd.to_numeric(df['Wavelength'], errors='coerce')
df['Zero Air Mass'] = pd.to_numeric(df['Zero Air Mass'], errors='coerce')
df['Earth Surface'] = pd.to_numeric(df['Earth Surface'], errors='coerce')
print(df.dtypes)

# Visible Range Analysis
visible_df = df[(df['Wavelength'] >= 400) & (df['Wavelength'] <= 700)]
visible_intensity_sum = visible_df['Earth Surface'].sum()
total_intensity_sum = df['Earth Surface'].sum()
visible_fraction = visible_intensity_sum / total_intensity_sum
print("Fraction of light power in the visible range at Earth's surface:", visible_fraction)

# Initial Plotting
plt.figure(figsize=(10, 5))
plt.plot(df['Wavelength'].values, df['Zero Air Mass'].values, label='Zero Air Mass')
plt.plot(df['Wavelength'].values, df['Earth Surface'].values, label='Earth Surface')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Irradiance (W*m-2*nm-1)')
plt.title('Solar Spectrum')
plt.legend()
plt.savefig('solar_spectrum.png')

# Photon Energy Calculations
support_df = df.copy()
support_df['Wavelength'] = support_df['Wavelength'] * 1e-9  # Convert to meters
support_df['Photon Energy'] = PLANCK_CONSTANT * SPEED_OF_LIGHT / support_df['Wavelength']
support_df['Zero Air Mass'] = support_df['Zero Air Mass'] / (AVOGADRO_NUMBER * support_df['Photon Energy'])
support_df['Earth Surface'] = df['Earth Surface'] / (AVOGADRO_NUMBER * support_df['Photon Energy'])

# as before make an analytics of what amount of photons are in the visible region at earth surface
visible_df = support_df[(support_df['Wavelength'] >= 400e-9) & (support_df['Wavelength'] <= 700e-9)]
total_intensity_sum = support_df['Earth Surface'].sum()
visible_intensity_sum = visible_df['Earth Surface'].sum()
visible_fraction = visible_intensity_sum / total_intensity_sum
print("Fraction of photon power in the visible range at Earth's surface:", visible_fraction)


# Make a second plot with the corrected units
plt.figure(figsize=(10, 5))
plt.plot(support_df['Wavelength'].values, support_df['Zero Air Mass'].values, label='Zero Air Mass')
plt.plot(support_df['Wavelength'].values, support_df['Earth Surface'].values, label='Earth Surface')

plt.xlabel('Wavelength (m)')
plt.ylabel('Irradiance (Einstein*s-1*m-1)')
plt.title('Solar Spectrum')
plt.legend()
# Save the plot
plt.savefig('einstein.png')

import numpy as np  # Import numpy for logarithmic calculations

# Plotting the absorbance
plt.figure(figsize=(10, 5))

# Calculate the absorbance
absorbance = -np.log10(df['Earth Surface'].values / df['Zero Air Mass'].values)

# Plot the absorbance as a function of wavelength
plt.plot(df['Wavelength'].values, absorbance)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.title('Atmospheric Absorbance')

# Save the plot
plt.savefig('absorbance_corrected.png')

