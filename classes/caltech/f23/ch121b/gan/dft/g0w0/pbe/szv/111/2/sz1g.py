import os
from pyscf.lib import chkfile
from pyscf.pbc.lib.chkfile import load_cell
from pyscf.pbc import scf, gw

script_dir = os.path.dirname(__file__)  # Gets the directory where the script is located
# Set the file path for the checkpoint file
file_path = os.path.join(script_dir, 'previous.chk')

# Load the cell object
cell = load_cell(file_path)
cell.build()

# Load the checkpoint file
scf_data = chkfile.load(file_path, 'scf')


# Create and populate the KRKS object
kpts = cell.make_kpts([1,1,1])
kmf = scf.KRKS(cell).density_fit()
kmf.kpts = kpts
kmf.__dict__.update(scf_data)


# Default is AC frequency integration
mygw = gw.KRGW(kmf)
mygw.kernel()
print("KRGW energies =", mygw.mo_energy)

# # Extracting HOMO and LUMO energies
nocc = cell.nelectron // 2  # Number of occupied orbitals

gw_homo = max(mygw.mo_energy[0][:nocc])
gw_lumo = min(mygw.mo_energy[0][nocc:])
gw_gap = gw_homo - gw_lumo
print("HOMO energy:", gw_lumo)
print("LUMO energy:", gw_homo)
print("Band gap at Gamma point [eV]:", gw_gap *27.211)
