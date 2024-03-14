import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random

class MonteCarlo():

    def __init__(self, x, alpha, n_iter):
        self.x = x  # Fraction of interacting spins
        self.beta = alpha  # Inverse temperature
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
 


        
   
        self.oldstate = np.copy(state)
        self.newstate = np.copy(state)

        self.initial = np.copy(state)
        self.E = 0
        for i in range(40):
            for j in range(40):
                self.E += self.local_hamiltonian(i, j, self.oldstate)
        self.E /= 2 #overcounting
        self.dE = 0 #placeholder

    def plot_state(self, state, n_iter, title="Lattice State"):
        plt.figure(figsize=(6, 6))
        plt.title(title)
        plt.imshow(state, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Spin Value')
        # Dynamically generate the filename using the iteration number
        filename = f"lattice_state_{n_iter}.png"
        plt.savefig(filename)
        plt.close()  # Close the figure after saving to free up memory
        return
        


    
    def run_iter(self):
        # first consider the flips
        for i in range(self.ndim):
            for j in range(self.ndim):
                # consider all possible flips only for the interacting spins
                if abs(self.oldstate[i][j]) == 1:
                    self.flip(i, j)
                    # calculate the change in energy associated
                    self.dE = self.local_hamiltonian(i, j, self.newstate) - self.local_hamiltonian(i, j, self.oldstate)
                    self.dE *= self.beta
                    # except or reject the switch based off of the metropolis-hastings algorithm
                    if self.update():
                        self.accept()
                    self.newstate = np.copy(self.oldstate)
                    
        # now consider the swaps
        for i in range(self.ndim):
            for j in range(self.ndim):
                # no need to check whether the spins are interacting or not
                neighbor = self.swap(i, j)
                # calculate the change new energy associated
                new_energy = self.local_hamiltonian(i, j, self.newstate) + self.local_hamiltonian(neighbor[0], neighbor[1], self.newstate)
                old_energy = self.local_hamiltonian(i, j, self.oldstate) + self.local_hamiltonian(neighbor[0], neighbor[1], self.oldstate)
                self.dE = new_energy - old_energy
                self.dE *= self.beta
                # except or reject the switch based off of the metropolis-hastings algorithm
                if self.update():
                    self.accept()
                self.newstate = np.copy(self.oldstate)       
        return

        
                

            
    
    def flip(self, i, j):
        ################################################################
        # THIS FUNCTION SHOULD PERFORM A FLIP AT INDEX i AND j, AND    #
        # COMPUTE THE ASSOCIATED ENERGY CHANGE WITH THiS FLIP          #
        ################################################################
        # changed the sign at index i, j
        self.newstate[i][j] = -self.newstate[i][j]
        return



    def swap(self, i, j):
        ################################################################
        # THIS FUNCTION SHOULD PERFORM A SWAP OF INDEX i AND j, AND    #
        # COMPUTE THE ASSOCIATED ENERGY CHANGE WITH THiS SWAP          #
        ################################################################
        # Define offsets for the 8 neighbors: top, bottom, left, right, and the 4 diagonals
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        # compute their indices given the periodic boundary conditions, which means taking mod self.ndim
        neighbors = [((i + di) % self.ndim, (j + dj) % self.ndim) for di, dj in offsets]
        # choose a random neighbor using random.choice
        neighbor = random.choice(neighbors)
        # swap the spins
        self.newstate[i][j], self.newstate[neighbor[0]][neighbor[1]] = self.newstate[neighbor[0]][neighbor[1]], self.newstate[i][j]
        # returned the index of the neighbor that was chosen
        return neighbor


    
    def update(self):

        ################################################################
        # THIS FUNCTION SHOULD CHECK WHETHER OR NOT A GIVEN OPERATION  #
        # SHOULD BE ACCEPTED. THIS IS YOUR METRPOLIS-HASTING ALGORITHM #
        ################################################################
        
        condition = False
        if self.dE <= 0 or np.random.rand() <= np.exp(-self.dE):
            condition = True
        # REGARDLESS OF THE OUTCOME ABOVE, YOU STILL NEED TO INITILIASE
        # YOUR NEW STATE FOR THE NEXT ITERATION
        # self.newstate = np.copy(self.oldstate)
        return condition
        


    def accept(self):
        ################################################################
        # THIS FUNCTION SHOULD UPDATE YOUR OLDSTATE AND ENERGY GIVEN   #
        # A CHANGE WAS ACCEPTED                                        $
        ################################################################
        self.oldstate = np.copy(self.newstate)
        self.E += self.dE
        return


    
    def local_hamiltonian(self, i, j, state):
        E = 0
        ################################################################
        # THIS FUNCTION SHOULD CALCULATE THE CHANGE IN THE HAMILTONIAN #
        # LOCALLY AROUND i and j. REMEMBER TO ACCOUNT FOR PERIOD BOUN- #
        # -DARY CONDITIONS.                                            #
        ################################################################
        # we only add a contribution to the energy if the spin is interacting
        if abs(state[i][j]) == 1:
            # Define offsets for the 4 neighbors: top, bottom, left, right, 
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            # compute their indices given the periodic boundary conditions, which means taking mod self.ndim
            neighbors = [((i + di) % self.ndim, (j + dj) % self.ndim) for di, dj in offsets]
            # check if the neighbor is in interacting spin
            for neighbor in neighbors:
                if abs(state[neighbor[0]][neighbor[1]]) == 1:
                    E += -state[i][j] * state[neighbor[0]][neighbor[1]]       
        return E
    

    def get_energy(self):
        return self.E

    def get_m(self):
        ################################################################
        # THiS FUNCTION SHOULD CALCULATE THE AVERAGE MAGNETISATION OF  #
        # YOUR SYSTEM AT A GIVEN ITERATION.                            #
        ################################################################
        magnetization = (1/self.Na) * np.sum(self.oldstate)
        return magnetization
    
    def run(self):
        self.initial = np.copy(self.oldstate)

        self.ms = [self.get_m()]
        self.Es = [self.get_energy()]

        for i in range(self.ni):
            self.run_iter()
            if i % 200 == 0:
                self.plot_state(self.oldstate, i)            
            
            self.Es.append(self.get_energy())
            self.ms.append(self.get_m())
        
        self.final = self.oldstate

MC = MonteCarlo(0.75,3,1e3)

MC.run()

print(MC.initial)
print(MC.final)
print(MC.ms)
# plot the energies vs the number of iterations
plt.figure()
plt.plot(MC.Es)
plt.xlabel('Iterations')
plt.ylabel('Energy')
plt.title('Energy vs Iterations')
plt.savefig('Energy_vs_Iterations.png')
