import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.optimize import root_scalar
import csv
import os
import shutil
from glob import glob

def objective_function(m, *args):
    # Objective function for which we want to find the root.

    # m: variable to be solved for that satisfies F(m) = 0
    # args: additional arguments to the function
         # a = args[0]: the alpha parameter
    
    # return: F(m) where you have rearranged the equation in the form F(m) = 0
    # If using Newton method, return: F(m) and F'(m)
    
    a = args[0]

    return 

# TODO: Fill in with answers from question 1
def calc_mu(x,m,a):
    # Calculate the chemical potential for a given x,m,a

    # x: the fraction of A particles
    # m: the relative magnetization
    # a: the alpha parameter

    # return: mu

    # TODO: fill in the return line to return mu
    return

# TODO: Fill in with answers from question 1
def calc_g(x,m,a):
    # Calculate the non-dimensional free energy per particle for a given x,m,a

    # x: the fraction of A particles
    # m: the relative magnetization
    # a: the alpha parameter

    # return: g

    # TODO: fill in the return line to return g
    return

def write_data_to_file(a, x_vec, m_vec, mu_vec, g_vec, dir):
    # Write data to a file

    # a: the alpha parameter
    # x_vec: the vector containing the fraction of A particles
    # m_vec: the vector containing the relative magnetizations
    # mu_vec: the vector containing the chemical potentials
    # g_vec: the vector containing the non-dimensional free energies

    # return: None

    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)

    # Concatenate the vectors into an array
    data = np.array([x_vec, m_vec, mu_vec, g_vec])

    # Write the data to a file
    np.savetxt(f'data_{a:.2f}.dat', data.T, delimiter=' ')

def generate_data_a(a):
    # Generate data for 2(i) and 2(ii) for a given alpha value. You should
    # use the functions you defined above to calculate the data.

    # a: the alpha value

    # return: x_vec, m_vec, mu_vec, g_vec (vectors containing the data)


    # Grid in x to calculate values for
    x_vec = np.linspace(0.9999,0.0001,50000)
    n = len(x_vec)

    # Initialize vectors to store the results
    m_vec = np.zeros(n)
    mu_vec = np.zeros(n)
    g_vec = np.zeros(n)

    # Set initial value of m to 1.5
    m0 = 1.5

    # TODO: Loop over x and solve for m, mu, and g
    for i,x in enumerate(x_vec):

        # TODO: solve for m using objective_function(m, *args) using m0 as guess
        m = None

        # TODO: calculate mu using calc_mu(x,m,a)
        mu = None

        # TODO: calculate g using calc_g(x,m,a)
        g = None

        # Update initial guess for next iteration
        m0 = m

        # Store the results in the vectors
        m_vec[i] = m
        mu_vec[i] = mu
        g_vec[i] = g

    return x_vec, m_vec, mu_vec, g_vec

def calc_data_2d_i_ii(outdir):
    a_vec = np.linspace(1,9,9)
    n = len(a_vec)

    for j,a in enumerate(a_vec):
        print(f'Calculating results for alpha = {a} ({j}/{n})...')

        # Generate the data for a given alpha
        x_vec, m_vec, mu_vec, g_vec = generate_data_a(a)

        # Write the data to a file
        write_data_to_file(a, x_vec, m_vec, mu_vec, g_vec, outdir)

# TODO: Fill in function to plot the data for part 2d (i, ii, and iii)
def plotting_2d_i_ii_iii(indir, outdir):
    # Plot the data for part 2d (i, ii, and iii)

    # indir: the directory containing the data files
    # outdir: the directory to save the figures

    # return: None

    # Make figures dir if it doesn't exist
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    # Get the data files
    files = glob(f'{indir}*.dat')
    alphas = [float(f.split('_')[-1].split('.dat')[0]) for f in files]
    alphas = np.array(alphas)
    indices = np.argsort(alphas)
    files = np.array(files)[indices]
    alphas = alphas[indices]

    # TODO: Generate plot for 2d (i) and save to file
    plt.figure(1,figsize=(5,4),dpi=300)
    for a,f in zip(alphas,files):
        
        
        pass
    plt.xlabel(r'$x$')
    plt.ylabel(r'$m$')
    plt.title(r'$m$ vs $x$ for different $\alpha$')
    plt.legend(loc='upper left',bbox_to_anchor=(1.01, 1.0), title=r"$\alpha=$")
    plt.tight_layout()
    plt.savefig(outdir+'m-x.png',dpi=300)


    # TODO: Generate plot for 2d (ii) and save to file
    plt.figure(2,figsize=(5,4),dpi=300)
    for a,f in zip(alphas,files):
        
        
        pass
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$g$')
    plt.title(r'$g$ vs $\mu$ for different $\alpha$')
    plt.legend(loc='upper left',bbox_to_anchor=(1.01, 1.0), title=r"$\alpha=$")
    plt.tight_layout()
    plt.savefig(outdir+'g-mu.png',dpi=300)

    # TODO: Generate plot for 2d (iii) and save to file
    plt.figure(3,figsize=(5,4),dpi=300)

    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$g$')
    plt.title(r'$g$ vs $\mu$ for $\alpha=9$')
    plt.tight_layout()
    plt.savefig(outdir+'g-mu-9.png',dpi=300)

# TODO: Fill in function to calculate spinodal and binodal
def calc_data_2d_iv():
    a_vec = np.linspace(1,9,20)
    n = len(a_vec)

    if os.path.exists('spinodals_binodal.dat'):
        os.remove('spinodals_binodal.dat')
    if os.path.exists('lambda_line.dat'):
        os.remove('lambda_line.dat')

    for j,a in enumerate(a_vec):
        print(f'Calculating results for alpha = {a} ({j}/{n})...')

        x_vec, m_vec, mu_vec, g_vec = generate_data_a(a)

        # TODO: Determine the spinodal points
        #   - The spinodals will be where the curves reverse direction on
        #     the g vs mu plot, meaning that mu goes from increasing to
        #     decreasing or vice versa.
        condition = False
        for i in range(1,len(mu_vec)-1):

            # TODO: check if the direction reversed (first time) and save 
            #       the first spinodal
            mu_spinodal_1 = None

            # TODO: check if the direction reversed (second time) and save 
            #       the second spinodal. Change condition to True if found
            mu_spinodal_2 = None


            if condition:
                print(f'Found spinodal points for alpha = {a:.4f}: mu_spinodal_1 = {mu_spinodal_1:.4f}, mu_spinodal_2 = {mu_spinodal_2:.4f}')
                break
        
        # If a spinodal is not found, calculate critical point and continue
        #    - Note that you should find spinodals for all alphas > 3
        #    - For 1 < alpha < 3, there are no spinodals or binodal, but
        #      there will be a critical point which you can find by other means
        if not condition:
            mu_spinodal_1 = np.nan
            mu_spinodal_2 = np.nan
            # TODO: Determine the critical point (analytically or numerically)
            #   - You should be able to find the critical point for 1<alpha<3
            mu_critical = None

            # Write the lambda line data for the current alpha to a file
            with open('lambda_line.dat', 'a+') as f:
                writer = csv.writer(f, delimiter=' ', lineterminator='\n')
                writer.writerow([a, mu_critical])
        else:
            # TODO: Determine the binodal (crossing of g vs mu)
            #   - You should find where the ordered and disordered branches cross
            #   - You might find it helpful to use splines and determine crossing,
            #     however, there are many many ways to do this!
            mu_binodal = None

            # Write the phase diagram data for the current alpha to a file
            with open('spinodals_binodal.dat', 'a+') as f:
                writer = csv.writer(f, delimiter=' ', lineterminator='\n')
                writer.writerow([a, mu_spinodal_1, mu_spinodal_2, mu_binodal])

# TODO: Fill in function to plot the phase diagram
def plotting_2d_iv():
    # Plot the phase diagram


    # Load the spinodals and binodal data
    data = np.loadtxt('spinodals_binodal.dat', delimiter=' ')
    a = data[:,0]
    mu_spinodal_1 = data[:,1]
    mu_spinodal_2 = data[:,2]
    mu_binodal = data[:,3]

    # Load the lambda line data
    data = np.loadtxt('lambda_line.dat', delimiter=' ')
    a_critical = data[:,0]
    mu_critical = data[:,1]

    # TODO: Generate plot for 2d (iv) and save to file
    #   - Plot the different parts of the phase diagram
    #   - both spinodals, binodal, lambda line
    plt.figure(4,figsize=(5,4),dpi=300)


    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$\alpha$')
    plt.title(r'Phase Diagram')
    plt.legend(loc='upper right',frameon=False, fancybox=False)
    plt.tight_layout()
    plt.savefig('phase_diagram.png',dpi=300)


if __name__ == "__main__":

    datadir = 'data/'
    figuredir = 'figures/'

    # UNCOMMENT to run calculation code for parts 2d (i) and 2 (ii)
    # calc_data_2d_i_ii(outdir=datadir)

    # UNCOMMENT to run plotting code for part 2d (i, ii,and iii)
    # plotting_2d_i_ii_iii(indir=datadir, outdir=figuredir)

    # UNCOMMENT to run calculation code for part 2d (iv)
    # calc_data_2d_iv()

    # UNCOMMENT to run plotting code for part 2d (iv)
    # plotting_2d_iv()

