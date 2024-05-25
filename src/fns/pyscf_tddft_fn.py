from pyscf import tddft

def pyscf_td(td_method, mf):
    '''Returns the pyscf td object that I want.'''
    # set of the common variables
    n_orbitals = mf.mol.nao_nr()
    n_occupied = mf.mol.nelectron//2
    n_virtual = n_orbitals - n_occupied

    if td_method == 'dtda':
        td = tddft.dTDA(mf)
    elif td_method == 'drpa':
        td = tddft.dRPA(mf)

    td.nstates = n_occupied*n_virtual
    e, xy = td.kernel()
    # # Make a fake Y vector of zeros
    # td_xy = list()
    # if td == 'drpa':
    #     for x, y in td.xy:
    #         td_xy.append((x, y))
    # if td == 'dtda':
    #     for x, y in td.xy:
    #         td_xy.append((x, 0*x))
    # td.xy = td_xy
    return e