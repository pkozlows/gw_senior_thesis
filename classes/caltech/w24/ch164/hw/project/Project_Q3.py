import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class MonteCarlo():

    def __init__(self, x, beta, n_iter):
        self.x = x  # Fraction of interacting spins
        self.beta = beta  # Inverse temperature
        self.ni = int(n_iter)  # Number of iterations
        self.N = 1600
        self.ndim = int(np.sqrt(self.N))  # Assuming a square lattice
        self.Na = int(self.N*x)
        self.Nb = int(self.N - self.Na)

        np.random.seed(24)

        # Initialize a two-dimensional square lattice
        state = np.zeros((self.ndim, self.ndim), dtype=int)

        # Randomly choose Na positions in the lattice for the interacting spins
        interacting_indices = np.random.choice(self.N, self.Na, replace=False)

        # Convert linear indices to 2D indices and assign s = +/- 1 to these positions
        for index in interacting_indices:
            i, j = divmod(index, self.ndim)
            state[i, j] = np.random.choice([-1, 1])
 
        np.random.seed(24)

        def plot_state(state, title="Lattice State"):
            plt.figure(figsize=(6, 6))
            plt.title(title)
            plt.imshow(state, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Spin Value')
            plt.savefig('lattice.png')
        
        plot_state(state)
   
        self.oldstate = np.copy(state)
        self.newstate = np.copy(state)

        self.initial = np.copy(state)

        self.E = 0
        for i in range(40):
            for j in range(40):
                self.E += self.local_hamiltonian(i, j, self.oldstate)
        self.E /= 8 #overcounting
        self.dE = 0 #placeholder
    
    
    def run_iter(self):
        # first consider all possible flips
        for i, Na in enumerate(self.oldstate):
            # check if the absolute value of the spin is equal to 1
            if abs(Na) == 1:
                Working on monte Carlo
                self.flip(i, j)
        # now consider all possible swaps
        
                

            
        # YOU SHOULD HAVE TWO FOR LOOPS: ONE FOR FLIPS, THE OTHER FOR 
        # SWAPS. AFTER EACH FLIP AND SWAP, YOU SHOULD UPDATE YOUR STATE 

        return # NOTHING SHOULD NEED TO BE OUTPUT
    
    def flip(self, i, j):
        ################################################################
        # THIS FUNCTION SHOULD PERFORM A FLIP AT INDEX i AND j, AND    #
        # COMPUTE THE ASSOCIATED ENERGY CHANGE WITH THiS FLIP          #
        ################################################################
        # changed the sign at index i, j
        self.newstate[i][j] = -self.newstate[i][j]
        return


    def swap(self, i, j, nn):
        ################################################################
        # THIS FUNCTION SHOULD PERFORM A SWAP OF INDEX i AND j, AND    #
        # COMPUTE THE ASSOCIATED ENERGY CHANGE WITH THiS SWAP          #
        ################################################################
        # from the point of view of a single spin in the squire lattice, perform a swap of spins with one of its 8 nearest neighbors.
        self.newstate[i][j] = self.oldstate[nn]
        self.newstate[nn] = self.oldstate[i][j]
        return
    
    def update(self):

        ################################################################
        # THIS FUNCTION SHOULD CHECK WHETHER OR NOT A GIVEN OPERATION  #
        # SHOULD BE ACCEPTED. THIS IS YOUR METRPOLIS-HASTING ALGORITHM #
        ################################################################
        
        # REGARDLESS OF THE OUTCOME ABOVE, YOU STILL NEED TO INITILIASE
        # YOUR NEW STATE FOR THE NEXT ITERATION
        self.newstate = np.copy(self.oldstate)
        return # NOTHING SHOULD NEED TO BE OUTPUT
        


    def accept(self):
        ################################################################
        # THIS FUNCTION SHOULD UPDATE YOUR OLDSTATE AND ENERGY GIVEN   #
        # A CHANGE WAS ACCEPTED                                        $
        ################################################################
        return # NOTHING SHOULD NEED TO BE OUTPUT


    
    def local_hamiltonian(self, i, j, state):
        E = 0
        ################################################################
        # THIS FUNCTION SHOULD CALCULATE THE CHANGE IN THE HAMILTONIAN #
        # LOCALLY AROUND i and j. REMEMBER TO ACCOUNT FOR PERIOD BOUN- #
        # -DARY CONDITIONS.                                            #
        ################################################################
        
        return E
    

    def get_energy(self):
        return self.E

    def get_m(self):
        ################################################################
        # THiS FUNCTION SHOULD CALCULATE THE AVERAGE MAGNETISATION OF  #
        # YOUR SYSTEM AT A GIVEN ITERATION.                            #
        ################################################################
        
        return 
    
    def run(self):
        self.initial = np.copy(self.oldstate)

        self.ms = [self.get_m()]
        self.Es = [self.get_energy()]

        for i in range(self.ni):
            self.run_iter()
            self.Es.append(self.get_energy())
            self.ms.append(self.get_m())
        
        self.final = self.oldstate

MC = MonteCarlo(0.75,3,1e3)

MC.run()

print(MC.ibetatial)
print(MC.final)
print(MC.ms)