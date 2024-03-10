import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class MonteCarlo():

    def __init__(self, x, alpha, ni):
        self.x = x
        self.beta = alpha
        self.ni = int(ni)
        self.N = 1600
        Na = int(self.N*x)
        Nb = int(self.N - Na)

 
        np.random.seed(24)


        ######################################################
        # INSERT CODE TO OBTAIN INITIAL STATE OF YOUR SYSTEM $
        ######################################################
   
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
        ################################################################
        # THIS FUNCTION SHOULD RUN ONE SINGLE ITERATION OF YOUR MC SIM #
        ################################################################
        # YOU SHOULD HAVE TWO FOR LOOPS: ONE FOR FLIPS, THE OTHER FOR 
        # SWAPS. AFTER EACH FLIP AND SWAP, YOU SHOULD UPDATE YOUR STATE 

        return # NOTHING SHOULD NEED TO BE OUTPUT
    
    def flip(self, i, j):
        ################################################################
        # THIS FUNCTION SHOULD PERFORM A FLIP AT INDEX i AND j, AND    #
        # COMPUTE THE ASSOCIATED ENERGY CHANGE WITH THiS FLIP          #
        ################################################################
        return # NOTHING SHOULD NEED TO BE OUTPUT


    def swap(self, i, j):
        ################################################################
        # THIS FUNCTION SHOULD PERFORM A SWAP OF INDEX i AND j, AND    #
        # COMPUTE THE ASSOCIATED ENERGY CHANGE WITH THiS SWAP          #
        ################################################################
        return # NOTHING SHOULD NEED TO BE OUTPUT
    
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

print(MC.initial)
print(MC.final)
print(MC.ms)