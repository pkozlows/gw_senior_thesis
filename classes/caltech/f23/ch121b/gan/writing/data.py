import matplotlib.pyplot as plt

# Data
k_points = [1, 16, 27]
b3pw91 = {'gth-szv': [7.87, 8.94, None], 'gth-dzvp': [1.91, 2.62, 2.72]}
pbe = {'gth-szv': [5.50, 7.07, 7.19], 'gth-dzvp': [2.35, 3.02, 3.12]}
g0w0_pbe = {'gth-szv': [None, None, None], 'gth-dzvp': [2.48, 3.86, 4.198692831650493]}

# Set global parameters for font size
plt.rcParams.update({'font.size': 12})  # Adjust the number to change the global font size

# Plot for 'gth-szv'
plt.figure(figsize=(10, 6))
plt.plot(k_points, b3pw91['gth-szv'], marker='o', linestyle='--', color='blue', label='KS@B3PW91')
plt.plot(k_points, pbe['gth-szv'], marker='s', linestyle='--', color='red', label='KS@PBE')
plt.axhline(y=3.503, color='black', linestyle='-', label='Experimental: 3.503 eV')
plt.xlabel('# K Points', fontsize=14)  # Adjust font size for x-label
plt.ylabel('Band Gap (eV)', fontsize=14)  # Adjust font size for y-label
plt.title('GaN w/ Small Basis', fontsize=16)  # Adjust font size for title
plt.legend(fontsize=14)  # Adjust font size for legend
plt.xticks(fontsize=14)  # Adjust font size for x-ticks
plt.yticks(fontsize=14)  # Adjust font size for y-ticks
plt.grid(True)
plt.xlim(1, 27)
plt.savefig('band_gaps_szv.png')

# Plot for 'gth-dzvp'
plt.figure(figsize=(10, 6))
plt.plot(k_points, b3pw91['gth-dzvp'], marker='o', linestyle='--', color='blue', label='KS@B3PW91')
plt.plot(k_points, pbe['gth-dzvp'], marker='s', linestyle='--', color='red', label='KS@PBE')
plt.plot(k_points, g0w0_pbe['gth-dzvp'], marker='^', linestyle='-', color='green', label='G0W0@PBE')
plt.axhline(y=3.503, color='black', linestyle='-', label='Experimental: 3.503 eV')
plt.xlabel('# K Points', fontsize=14)
plt.ylabel('Band Gap (eV)', fontsize=14)
plt.title('GaN w/ Large Basis', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.xlim(1, 27)
plt.savefig('band_gaps_dzvp.png')
