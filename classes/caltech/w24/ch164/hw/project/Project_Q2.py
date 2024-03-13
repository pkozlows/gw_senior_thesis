import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.optimize import root_scalar
from scipy.interpolate import CubicSpline
import csv
import os
import shutil
from glob import glob

def objective_function(m, *args):
    # Objective function for which we want to find the root: m - \tanh(\alpha mx) = 0
    a = args[0]
    x = args[1]
    f = m - np.tanh(a * m * x)



    # m: variable to be solved for that satisfies F(m) = 0
    # args: additional arguments to the function
         # a = args[0]: the alpha parameter
            # x = args[1]: the fraction of A particles

    return f

def objective_function_prime(m, *args):
    # Derivative of the objective function: F'(m) = 1 - \alpha x \text{sech}^{2}(\alpha m x)
    a = args[0]
    x = args[1]
    f_prime = 1 - a * x * (1/np.cosh(a * m * x))**2

    # m: variable to be solved for that satisfies F(m) = 0
    # args: additional arguments to the function
         # a = args[0]: the alpha parameter
            # x = args[1]: the fraction of A particles

    return f_prime

def calc_mu(x, m, a):
    # Calculate the chemical potential for a given x, m, a

    mu = -1.0 * a * m**2 * x + (0.5 - m/2) * np.log(x * (0.5 - m/2)) + (m/2 + 0.5) * np.log(x * (m/2 + 0.5)) - np.log(1 - x)

    return mu



def calc_g(x,m,a):
    # Calculate the non-dimensional free energy per particle for a given x,m,a

    # x: the fraction of A particles
    # m: the relative magnetization
    # a: the alpha parameter

    # this is the equation that I want to turn into python code: g = -\frac{1}{2} \alpha (m x)^{2} - \mu x + \left( \frac{1 + m}{2}x \ln \frac{1 + m}{2}x + \frac{1 - m}{2}x \ln \frac{1 - m}{2}x + (1 - x) \ln (1 - x) \right)
    g = -0.5 * a * (m * x)**2 - calc_mu(x,m,a) * x + (0.5 * (1 + m) * x * np.log(0.5 * (1 + m) * x) + 0.5 * (1 - m) * x * np.log(0.5 * (1 - m) * x) + (1 - x) * np.log(1 - x))

    return g

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

    for i,x in enumerate(x_vec):

        m = root_scalar(objective_function, fprime=objective_function_prime, args=(a,x), x0=m0, method='newton').root


        mu = calc_mu(x,m,a)

        g = calc_g(x,m,a)

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

    plt.figure(1,figsize=(5,4),dpi=300)
    for a,f in zip(alphas,files):
        data = np.loadtxt(f)  # Assuming the data file format is compatible
        x_vec, m_vec = data[:,0], data[:,1]  # Adapt this line based on your actual data structure
        plt.plot(x_vec, m_vec, label=f'{a:.2f}')

    plt.xlabel(r'$x$')
    plt.ylabel(r'$m$')
    plt.title(r'$m$ vs $x$ for different $\alpha$')
    plt.legend(loc='upper left',bbox_to_anchor=(1.01, 1.0), title=r"$\alpha=$")
    plt.tight_layout()
    plt.savefig(outdir+'m-x.png',dpi=300)


    plt.figure(2,figsize=(5,4),dpi=300)
    for a,f in zip(alphas,files):
        data = np.loadtxt(f)  # Assuming the data file format is compatible
        mu_vec, g_vec = data[:,2], data[:,3]  # Adapt this line based on your actual data structure
        plt.plot(mu_vec, g_vec, label=f'{a:.2f}')
        
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$g$')
    plt.title(r'$g$ vs $\mu$ for different $\alpha$')
    plt.legend(loc='upper left',bbox_to_anchor=(1.01, 1.0), title=r"$\alpha=$")
    plt.tight_layout()
    plt.savefig(outdir+'g-mu.png',dpi=300)

    plt.figure(3,figsize=(5,4),dpi=300)
    a = 9
    data = np.loadtxt(f'data_{a:.2f}.dat')  # Assuming the data file format is compatible
    mu_vec, g_vec = data[:,2], data[:,3]  # Adapt this line based on your actual data structure
    plt.plot(mu_vec, g_vec, label=f'{a:.2f}')
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

    for j, a in enumerate(a_vec):
        print(f'Calculating results for alpha = {a} ({j}/{n})...')

        x_vec, m_vec, mu_vec, g_vec = generate_data_a(a)

        mu_spinodal_1 = None
        mu_spinodal_2 = None

        for i in range(1, len(mu_vec) - 1):
            if g_vec[i] < g_vec[i - 1] and g_vec[i] < g_vec[i + 1]:
                mu_spinodal_1 = mu_vec[i]
            if g_vec[i] > g_vec[i - 1] and g_vec[i] > g_vec[i + 1]:
                mu_spinodal_2 = mu_vec[i]

        if mu_spinodal_1 is not None and mu_spinodal_2 is not None:
            print(f'Found spinodal points for alpha = {a:.4f}: mu_spinodal_1 = {mu_spinodal_1:.4f}, mu_spinodal_2 = {mu_spinodal_2:.4f}')

        if mu_spinodal_1 is None or mu_spinodal_2 is None:
            print(f'No spinodal points found for alpha = {a:.4f}')
            
        # If a spinodal is not found, calculate critical point and continue
        # Critical point calculation (placeholder, implement your actual method)
        # This is a simplified approach; you may need numerical differentiation and finding extrema
        # make a function to calculate the critical point
        def calculate_critical_point(mu_vec, g_vec):
            # Calculate the first derivative of g with respect to mu
            dg = np.gradient(g_vec, mu_vec)
            
            # Calculate the second derivative of g with respect to mu
            d2g = np.gradient(dg, mu_vec)
            
            # Find the index of the element where the second derivative is closest to zero
            # This is a simple approximation and assumes a single critical point
            critical_index = np.argmin(np.abs(d2g))
            
            # Return the mu value corresponding to this critical point
            mu_critical = mu_vec[critical_index]
            return mu_critical
        mu_critical = calculate_critical_point(mu_vec, g_vec)
        if mu_critical is not None:
            print(f'Found critical point for alpha = {a:.4f}: mu_critical = {mu_critical:.4f}')
        else:
            print(f'No critical point found for alpha = {a:.4f}')

        # Binodal point calculation using spline interpolation
        # spline_g = UnivariateSpline(mu_vec, g_vec, s=0, k=4)  # Spline of g vs. mu
        # You may need two splines if you have two phases and need to find their intersection
        # mu_binodal = find_spline_intersection(spline_g1, spline_g2)  # Placeholder method

        # Output results to file, similar to your existing logic for lambda_line.dat and spinodals_binodal.dat

        #     # TODO: Determine the critical point (analytically or numerically)
        #     #   - You should be able to find the critical point for 1<alpha<3
        #     mu_critical = None

        #     # Write the lambda line data for the current alpha to a file
        #     with open('lambda_line.dat', 'a+') as f:
        #         writer = csv.writer(f, delimiter=' ', lineterminator='\n')
        #         writer.writerow([a, mu_critical])
        # else:
        #     # TODO: Determine the binodal (crossing of g vs mu)
        #     #   - You should find where the ordered and disordered branches cross
        #     #   - You might find it helpful to use splines and determine crossing,
        #     #     however, there are many many ways to do this!
        #     mu_binodal = None

        #     # Write the phase diagram data for the current alpha to a file
        #     with open('spinodals_binodal.dat', 'a+') as f:
        #         writer = csv.writer(f, delimiter=' ', lineterminator='\n')
        #         writer.writerow([a, mu_spinodal_1, mu_spinodal_2, mu_binodal])

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
    # plotting_2d_i_ii_iii(indir='', outdir=figuredir)

    # UNCOMMENT to run calculation code for part 2d (iv)
    calc_data_2d_iv()

    # UNCOMMENT to run plotting code for part 2d (iv)
    # plotting_2d_iv()

