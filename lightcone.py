
import numpy as npy
import healpy as hp
import numexpr as ne
import numba as nb

import lcmetric.utils as ut
import pyshtools as pysh

class Lightcone:

    # does not really use rho_bar, replaced by H
    matter={'rho_bar':None, 'delta':None, 'vw':None}


    # For iteration scheme
    metric_a={ 'a':None, }
    metric_dt = {'a':None, }
    metric_f = {'a':None, 'Hubble':None, 'Hubble_dt':None}

    sols = {'Phi':None, 'Pi':None}

    

    Hubble_0 = None
    Hubble = None
    a = None
    Hubble_dt = None

    def  __init__(self,  nlat, lmax = None, \
                  epsilon = 1e-6, grid = 'healpy', alm_iter = 30, \
                  depth = 3, n_vcycles = 10, npre = 10, npost = 10, verbose = True):
        """Initialize light-cone class
        Parameters
        ----------
        nlat : int
        Is NSIDE when using healpix

        lmax: int, optional
        By default is 2*NSIDE-1

        epsilon: float, has NOT been integrated to multigrid yet
        Leading filename with path of the cone file

        grid: str, optional
        grid type, default is healpy. Currently keep pysh grid support, considering completely
        remove it in the future

        alm_iter: int, optional
        Number of iterations when using healpy map2alm function

        depth: int, optional
        Multigrid V-cycle depth, usually 4-6

        n_vcycles: int, optional
        Number of vcycles

        npre: int, optional
        Number of pre-relaxation at each level in the v-cycle

        npost: int, optional
        Number of post-relaxation at each level in the v-cycle

        verbose: bool, optional
        If output error info for every step, can be expensive!
        Returns
        -------
        void
        """
        if(grid == 'GLQ' or grid == 'DH'):
            self.nlat = nlat
            self.nlon = self.nlat * 2
            self.Npix = self.nlat * self.nlon
            self.lmax = lmax
            self.grid = grid
            if(self.lmax == None):
                self.lmax = int(self.nlat / 2) - 1
            if(grid == 'GLQ'): self.extend = True
            else: self.extend = False

            self.lm = npy.array([[ [-l*(l+1) for m in range(self.lmax+1)] for l in range(self.lmax+1)],
                                 [ [-l*(l+1) for m in range(self.lmax+1)] for l in range(self.lmax+1)]])
        elif(grid == 'healpy'):
            self.nside = nlat
            self.Npix = 12* self.nside * self.nside
            self.lmax = lmax
            self.grid = grid
            self.alm_iter = alm_iter
            if(self.lmax == None):
                self.lmax = self.nside * 2 - 1
            self.lm = hp.Alm.getlm( \
                self.lmax, npy.array(range( int(1+self.lmax + 0.5 * self.lmax*(1+self.lmax)) )))
            self.lm = -self.lm[0] *(self.lm[0] + 1)
        else:
            raise ValueError('Grid type '+str(grid) + ' does not exist!')
        

        self.depth = depth
        self.n_vcycles = n_vcycles
        self.npre = npre
        self.npost = npost
        
        
        self.epsilon = epsilon

        self.verbose = verbose

        

    def to_tau(self, ntau, d = 0):
        """
        From mesh idx to radius
        """
        return self.tau_f + ntau / self.Ntau * (self.tau_i - self.tau_f) 
    # ---------Functions of time derivatives-----------------------
        
    def da_dt(self, ntau):
        return self.metric_a['a'] * self.Hubble_a

    def dPhi_dt(self, ntau, d):
        return (-2 / self.tau_hier[d][ntau] + 2*self.Hubble_hier[d][ntau]) * self.Pi_hier[d][ntau] \
            - (2 * self.Hubble_dt_hier[d][ntau] -2 * self.Hubble_hier[d][ntau]**2
               - 3 * 1.5 * self.Hubble_0**2 *self.Omega_m / self.a_hier[d][ntau]) \
            * self.Phi_hier[d][ntau] \
            + 1.5 * self.Hubble_0**2 * self.Omega_m / self.a_hier[d][ntau] * self.delta_hier[d][ntau] \
            - self.lm * self.Phi_hier[d][ntau] / self.tau_hier[d][ntau]**2  \
    + 3 * self.Hubble_0**2 * self.Omega_m / self.a_hier[d][ntau] * self.vw_hier[d][ntau]

    def est_f(self, ntau, d):
        """
        Estimate rhs for relaxation (removing linear terms on Phi, which is on the left)
        """
        return (-2 / self.tau_hier[d][ntau] + 2*self.Hubble_hier[d][ntau]) * self.Pi_hier[d][ntau] \
            + 1.5 * self.Hubble_0**2 * self.Omega_m / self.a_hier[d][ntau] * self.delta_hier[d][ntau] \
    + 3 * self.Hubble_0**2 * self.Omega_m / self.a_hier[d][ntau] * self.vw_hier[d][ntau]

    
    def dPi_dt(self, ntau, d):
        if (ntau == self.h_size[d] - 1):
            Omega_dot = (self.Phi_hier[d][ntau] + self.Phi_hier[d][ntau-2] - 2*self.Phi_hier[d][ntau-1]) \
                / (self.dtau_hier[d])**2
            Omega = -(self.Phi_hier[d][ntau] - self.Phi_hier[d][ntau-1]) / self.dtau_hier[d]
        else:
            Omega_dot = (self.Phi_hier[d][ntau+1] + self.Phi_hier[d][ntau-1] - 2*self.Phi_hier[d][ntau]) \
                / (self.dtau_hier[d])**2
            Omega = -(self.Phi_hier[d][ntau + 1] - self.Phi_hier[d][ntau-1]) / (2.0 * self.dtau_hier[d])

            
        return - Omega_dot - 3 * self.Hubble_hier[d][ntau] * Omega \
            - 2 * self.Hubble_hier[d][ntau] * self.Pi_hier[d][ntau] \
            - (2 * self.Hubble_dt_hier[d][ntau] + self.Hubble_hier[d][ntau]**2) * self.Phi_hier[d][ntau] \
            + 1.5 * self.Hubble_0**2 * self.Omega_m / self.a_hier[d][ntau] * self.vw_hier[d][ntau]
                       
    
    # --------------------updating other fields-----------
    def update_other_field(self, ntau):
        self.Hubble_a = self.Hubble_0 * self.metric_a['a']\
            * npy.sqrt(self.Omega_m * self.metric_a['a']**(-3) + self.Omega_L )
        self.Hubble_dt_a = self.Hubble_0**2 * self.metric_a['a']**2 * self.Omega_L \
            - self.Hubble_0**2 * self.Omega_m / (2*self.metric_a['a'])

        # self.Hubble = 0
        # self.Hubble_dt = 0
        
        self.metric_f['Hubble'][ntau] = self.Hubble_a
        self.metric_f['Hubble_dt'][ntau] = self.Hubble_dt_a

        
    #-----------Time advancing----------------------------------
    # RK2
    def rk2_time_advance_step(self, ntau, dtau):
        if(ntau == 0):
            return

        #----------------first phase-----------------
        for field in self.metric_a:
            self.metric_dt[field] = eval('self.d'+field+'_dt')(ntau)

        for field in self.metric_a:
            self.metric_a[field] += self.metric_dt[field] * dtau
            self.metric_f[field][ntau-1] = self.metric_f[field][ntau] + 0.5 * dtau * self.metric_dt[field] 

        self.update_other_field(ntau-1)

        
        #------------------second phase--------------
        ntau = ntau - 1

        for field in self.metric_a:
            self.metric_dt[field] = eval('self.d'+field+'_dt')(ntau)


        for field in self.metric_a:
            self.metric_f[field][ntau] += 0.5 * dtau * self.metric_dt[field] 
            self.metric_a[field] = self.metric_f[field][ntau].copy()

        self.update_other_field(ntau)


    def est_errors(self, d, indent = ''):
        """
        Print error information. It is a little expensive!
        """
        err = 0
        rel_err = 0
        mag = 0
        max_step = -1
        for step in reversed(range(1, self.h_size[d]-1)):
            #self.est_angle_lap(step, d)
            rhs = self.dPhi_dt(step, d) + self.rhs_hier[d][step]
            lhs = (self.Phi_hier[d][step + 1] + self.Phi_hier[d][step - 1] \
                   - 2.0 * self.Phi_hier[d][step ])/ self.dtau_hier[d]**2 
            if(self.grid=='DH2' or self.grid == 'DH'):
                bak = err
                real_err = npy.abs(pysh.expand.MakeGridDH(rhs - lhs, sampling = 2, extend = False))
                err = npy.max([real_err.max(), err])
                rel_err += npy.linalg.norm(real_err)**2
                mag += npy.linalg.norm(pysh.expand.MakeGridDH(lhs, sampling = 2, extend = False))**2
                if(err > bak): max_step = step
            elif(self.grid == 'healpy'):

                bak = err
                real_err = npy.abs(hp.alm2map(rhs - lhs, self.nside))
                err = npy.max([real_err.max(), err])
                rel_err += npy.linalg.norm(real_err)**2
                mag += npy.linalg.norm(hp.alm2map(lhs, self.nside))**2
                if(err > bak): max_step = step
                
        rel_err = npy.sqrt(rel_err / mag)
        for field in self.sols:
            if field == 'Phi':
                print(indent+'Max error for field '+field+' is '+str(err) + ' at step '+str(max_step))
                print(indent+'Relative error of the L2 norm is ' + str(rel_err))
                                    
    def relax(self, d, nsteps):
        while(nsteps > 0):
            nsteps -= 1
            for field in self.sols:
                #for step in  reversed(range(self.ntau_f + 1, self.Ntau)):
                for step in  range(self.h_size[d] - 2, 0, -1):
                    if field == 'Phi':
                        # Newton relax Phi
                        self.Phi_hier[d][step] = \
                            ((self.est_f(step, d) + self.rhs_hier[d][step]) \
                             - (self.Phi_hier[d][step + 1] + self.Phi_hier[d][step - 1]) / self.dtau_hier[d]**2 \
                            ) / (-2 / self.dtau_hier[d]**2 + self.lm / self.tau_hier[d][step]**2 \
                                 + 2.0 * \
                                 (self.Hubble_dt_hier[d][step] - self.Hubble_hier[d][step]**2
                                  - 1.5**2 * self.Hubble_0**2 *self.Omega_m / self.a_hier[d][step])) 
                          
                    elif field == 'Pi':
                        # Time integrate Pi
                        dt = eval('self.d'+field+'_dt')(step + 1, d)
                        self.Pi_hier[d][step] = self.Pi_hier[d][step+1] \
                            + dt * self.dtau_hier[d]
                        dt += eval('self.d'+field+'_dt')(step, d)
                        self.Pi_hier[d][step] = self.Pi_hier[d][step+1] \
                            + 0.5 * self.dtau_hier[d] * dt

            
    def generate_rl(self, d):
        for step in  range(self.h_size[d] - 2, 0, -1):
            self.rl_hier[d][step] = (self.Phi_hier[d][step + 1] + self.Phi_hier[d][step - 1] \
                 - 2.0 * self.Phi_hier[d][step ])/ (self.dtau_hier[d])**2                    \
                 - self.dPhi_dt(step, d)

        
    def update_Pi(self, d):
        """
        Since Pi has directly dependence on Phi dot, this will 
        generate a new Pi whenever the value of Phi 
        """
        for step in  range(self.h_size[d] - 2, 0, -1):
            dt = self.dPi_dt(step + 1, d)
            self.Pi_hier[d][step] = self.Pi_hier[d][step+1] \
                + dt * self.dtau_hier[d]
            dt += self.dPi_dt(step, d)
            self.Pi_hier[d][step] = self.Pi_hier[d][step+1] \
                + 0.5 * self.dtau_hier[d] * dt

    def update_rhs(self, d):
        for step in  range(self.h_size[d] - 2, 0, -1):
            self.rhs_hier[d][step] += (self.Phi_hier[d][step + 1] + self.Phi_hier[d][step - 1] \
                                 - 2.0 * self.Phi_hier[d][step ])/ (self.dtau_hier[d])**2\
                                 - self.dPhi_dt(step, d) - self.rl_hier[d][step]

    def clear_field(self, data):
        for d in range(self.depth):
            data[d].fill(0) 
            
    def MG(self):
        """
        V-cycle multigrid. Starting from relaxing the finest grid. Might be able to 
        get speed-up when using more complicaed cycles with proper stop critiria 
        """
        n_vcycles = self.n_vcycles
        while(n_vcycles >0):
            n_vcycles -= 1
            print('Starting v-cycle '+ str(self.n_vcycles - n_vcycles))

            if(self.depth == 1):
                upper_stroke = range(1)
            else:
                upper_stroke = range(self.depth - 1)
            for d in upper_stroke:
                self.relax(d, int(self.npre ))
                print('    At level '+str(d)+', after '+str( int(self.npre) )+' relaxations')
                if(d == 0 or self.verbose == True): self.est_errors(d, '    ')
                if(d < self.depth - 1):
                    self.generate_rl(d)

                    #self.hier_restrict(self.Pi_hier,d)
                    self.Pi_bd_restrict(d)
                    
                    self.hier_restrict(self.rl_hier,d)
                    self.hier_restrict(self.Phi_hier,d)
                    
                    self.update_Pi(d+1)
                    self.hier_restrict(self.rhs_hier, d)
                    self.update_rhs(d+1)
            self.clear_field(self.rl_hier)
            for d in range(self.depth - 1, 0, -1):
                self.relax(d, int(self.npost ) )
                print('    At level '+str(d)+', after '+str( int(self.npost ) )+' relaxations')
                if(self.verbose == True): self.est_errors(d, '    ') 
                if(d > 0):
                    self.hier_restrict(self.Phi_hier, d-1, self.rl_hier)
                    self.rl_hier[d] = - self.rl_hier[d] + self.Phi_hier[d]
                    self.hier_prolong(self.rl_hier, d)
                    self.Phi_hier[d-1] += self.rl_hier[d-1]
                    self.update_Pi(d-1)
            self.clear_field(self.rhs_hier)
            self.clear_field(self.rl_hier)
            
        self.relax(0, int(self.npre ))
            
    def init(self):
        """
        Time evolving scale factor to get a, H, and dH_dt data at every light-cone grid
        """
        #set-up time evolution of a and Hubble and Hubble_dt
        self.update_other_field(self.Ntau)
        for step in reversed(range(1, self.Ntau + 1)):
            self.rk2_time_advance_step(step, (self.tau_i - self.tau_f) / self.Ntau)

        #set-up the initial guess
        for step in  reversed(range(1, self.Ntau)):
            for field in self.sols:
                if(field == 'Phi'):
                    self.sols[field][step] = self.sols[field][self.Ntau] \
                        + (self.Ntau - step) / (self.Ntau) \
                        * (self.sols[field][0] - self.sols[field][self.Ntau])

    # Only restricing boundary data for Pi
    # used by integration
    def Pi_bd_restrict(self, d):
        if( d == self.depth - 1):
            return
        self.Pi_hier[d+1][0] = self.Pi_hier[d][0]
        if(self.h_size[d] % 2 == 0):
            self.Pi_hier[d+1][-1] = self.Pi_hier[d][-2]
        else:
            self.Pi_hier[d+1][-1] = self.Pi_hier[d][-1]
        

    # Restricting data from depth d to d + 1 (fine to coarse)
    def hier_restrict(self, data, d, tdata = None):
        if( d == self.depth - 1):
            return
        if(tdata == None): tdata = data
        tdata[d+1][1:-1] = 0.5 * data[d][2:-2:2] + 0.25 * (data[d][1:-3:2] + data[d][3:-1:2])
        tdata[d+1][0] = data[d][0]
        if(self.h_size[d] % 2 == 0):
            tdata[d+1][-1] = data[d][-2]
        else:
            tdata[d+1][-1] = data[d][-1]
        
        
    # Prolonging data from depth d to d-1 (coarse to fine)
    def hier_prolong(self, data, d, tdata = None):
        if( d == 0):
            return
        if(tdata == None): tdata = data
        tdata[d-1][2:-2:2] = data[d][1:-1]
        tdata[d-1][1:-1:2] = 0.5 * (data[d][0:-1] + data[d][1:])
        tdata[d-1][0] = data[d][0]
        # if(self.h_size[d-1] % 2 == 0):
        #     tdata[d-1][-2] = data[d][-1]
        # else:
        #     tdata[d-1][-1] = data[d][-1]

    def to_alm(self, data, array_data = True):
        if(self.grid == 'GLQ' or self.grid =='DH' or self.grid == 'DH2'):
            if(array_data == True):
                return npy.array( [pysh.expand.SHExpandDH( \
                                                           data[n], sampling = 2) \
                                   for n in range(len(data))])
            else:
                return pysh.expand.SHExpandDH(data, sampling = 2) 
                                   
        else:
            if(array_data == True):
                return npy.array( [hp.sphtfunc.map2alm( \
                                                        data[n], lmax = self.lmax, iter = self.alm_iter) \
                                   for n in range(len(data))])
            else:
                return hp.sphtfunc.map2alm( \
                                     data, lmax = self.lmax, iter = self.alm_iter)

    def to_real(self, data, array_data = True):
        if(self.grid == 'GLQ' or self.grid =='DH' or self.grid == 'DH2'):
            raise ValueError('Stop supporting DH grids!')
        else:
            if(array_data == True):
                return npy.array( [hp.sphtfunc.alm2map( \
                                                        data[n], lmax = self.lmax) \
                                   for n in range(len(data))])
            else:
                return hp.sphtfunc.alm2map( \
                                     data, lmax = self.lmax)

        
    def build_hier(self, alm_form = False):

        # Initializing finest hiers
        self.Phi_hier = [None] * self.depth
        self.Pi_hier = [None] * self.depth
        self.delta_hier = [None] * self.depth
        self.vw_hier = [None] * self.depth
        self.a_hier = [None] * self.depth
        self.Hubble_hier = [None] * self.depth
        self.Hubble_dt_hier = [None] * self.depth
        self.lm_hier = [None] * self.depth
        self.tau_hier = [None] * self.depth
        self.dtau_hier = [None] * self.depth
        self.h_size = [None] * self.depth

        self.rhs_hier = [None] * self.depth
        self.rl_hier = [None] * self.depth


        if(alm_form == True):
            self.Phi_hier[0] = self.sols['Phi']
            self.Pi_hier[0] = self.sols['Pi']
        else:
            self.Phi_hier[0] = self.to_alm(self.sols['Phi'])
            self.Pi_hier[0] = self.to_alm(self.sols['Pi'])

        
        self.delta_hier[0] = self.to_alm(self.matter['delta'])
        self.vw_hier[0] = self.to_alm(self.matter['vw'])

        self.a_hier[0] = self.metric_f['a']
        self.Hubble_hier[0] = self.metric_f['Hubble']
        self.Hubble_dt_hier[0] = self.metric_f['Hubble_dt']


        if(self.grid == 'GLQ' or self.grid =='DH' or self.grid == 'DH2'):
            self.rhs_hier[0] = npy.zeros(self.Phi_hier[0].shape)
            self.rl_hier[0] = npy.zeros(self.Phi_hier[0].shape)

        else:
            self.rhs_hier[0] = npy.zeros(self.Phi_hier[0].shape, dtype = complex)
            self.rl_hier[0] = npy.zeros(self.Phi_hier[0].shape, dtype = complex)

        
        # Allocating space for hiers
        # and initializing data through hier_restrict
        self.h_size[0] = self.Ntau  + 1
        n = self.h_size[0] #h_size includes both ends
        self.h_size[0] = n
        for d in range(self.depth-1):
            if(n %2 == 0):
                n = int(n /2)
            else:
                n = int((n + 1) / 2)
                
            self.h_size[d+1] = n

            if(self.grid == 'GLQ' or self.grid == 'DH' or self.grid == 'DH2'):
                self.Phi_hier[d+1] = npy.zeros((n, 2, self.lmax+1, self.lmax+1))
                self.hier_restrict(self.Phi_hier, d)
                
                self.Pi_hier[d+1] = npy.zeros((n, 2, self.lmax+1, self.lmax+1))
                
                self.delta_hier[d+1] = npy.zeros((n, 2, self.lmax+1, self.lmax+1))
                self.hier_restrict(self.delta_hier, d)
                
                self.vw_hier[d+1] = npy.zeros((n, 2, self.lmax+1, self.lmax+1))
                self.hier_restrict(self.vw_hier, d)
                
                self.rhs_hier[d+1] = npy.zeros((n, 2, self.lmax+1, self.lmax+1))
                self.rl_hier[d+1] = npy.zeros((n, 2, self.lmax+1, self.lmax+1))
            else:
                self.Phi_hier[d+1] = npy.zeros((n, hp.Alm.getsize(self.lmax)), dtype = complex)
                self.hier_restrict(self.Phi_hier, d)
                
                self.Pi_hier[d+1] = npy.zeros((n, hp.Alm.getsize(self.lmax)), dtype = complex)
                
                self.delta_hier[d+1] = npy.zeros((n, hp.Alm.getsize(self.lmax)), dtype = complex)
                self.hier_restrict(self.delta_hier, d)
                
                self.vw_hier[d+1] = npy.zeros((n, hp.Alm.getsize(self.lmax)), dtype = complex)
                self.hier_restrict(self.vw_hier, d)
                
                self.rhs_hier[d+1] = npy.zeros((n, hp.Alm.getsize(self.lmax)), dtype = complex)
                self.rl_hier[d+1] = npy.zeros((n, hp.Alm.getsize(self.lmax)), dtype = complex)

                

            self.a_hier[d+1] = npy.zeros(n)
            self.hier_restrict(self.a_hier, d)
            
            self.Hubble_hier[d+1] = npy.zeros(n)
            self.hier_restrict(self.Hubble_hier, d)
            
            self.Hubble_dt_hier[d+1] = npy.zeros(n)
            self.hier_restrict(self.Hubble_dt_hier, d)
            

        for d in range(self.depth):
            self.tau_hier[d] = npy.array([self.to_tau(i, d) for i in range(self.h_size[d])])
            self.dtau_hier[d] = (self.tau_i - self.tau_f) / self.Ntau * 2**d

                
    #-------------Initializations------------------------------------

    def Phi_Pi_gen(self, delta, vw, dvw_dt, H, a, r):
        Pi_zel = -1.5 * H * vw

        PPi_zel = (Pi_zel + vw * H + dvw_dt) * 1.5*H
        
        
        H0 = self.Hubble_0
        rhs = r**2 * \
            (1.5 * H0**2 * self.Omega_m / a * delta \
             - (2 * Pi_zel + r * PPi_zel) / r )
        self.rhs = self.to_alm(rhs, array_data = False)

        self.coeffs = 2 * r**2 * (1.5 * self.Hubble_0**2 * self.Omega_m / a )
        
        self.Phi_zel = self.rhs / (self.coeffs + self.lm)

        self.Pi_zel = self.to_alm(Pi_zel, array_data = False)

        
        return self.Phi_zel, self.Pi_zel

    def init_from_matter(self, z_i_in, r_max_in, delta, vw, Params, \
                        z_f_in, r_min_in):
        """
        Initialize from delta and vw fields only from Zel' approximation
        Still under construction
        """
        self.r_max = r_max_in;

        self.tau_i = r_max_in;

        self.a_f = 1 / (1 + z_f_in)

        self.tau_f = r_min_in

        self.Ntau = delta.shape[0] - 2
        
        #self.ntau_f = int(npy.round(self.tau_f / (self.tau_i / self.Ntau)))

        for field in self.metric_f:
            self.metric_f[field] = npy.zeros(self.Ntau + 1)

        #descritation index ranging from 0 to Ntau
        if(self.grid == 'GLQ' or self.grid == 'DH' or self.grid == 'DH2'):    
            for field in self.sols:
                self.sols[field] = npy.zeros((self.Ntau + 1, 2, self.lmax+1, self.lmax+1))
        else:
            for field in self.sols:
                self.sols[field] = npy.zeros((self.Ntau + 1, hp.Alm.getsize(self.lmax)), dtype = complex)
            

        self.matter['delta'] = delta[0:self.Ntau+1].copy()
        self.matter['vw'] = vw[0:self.Ntau+1].copy()
        # Need to substract extra 2 padding grid 


        
        
        #self.sols['Phi'][-1] = Phi_i_in.copy()
        #self.sols['Pi'][-1] = Pi_i_in.copy()


        #self.sols['Phi'][0] = Phi_f_in.copy()
        #self.sols['Pi'][0] = Pi_f_in.copy()
        #self.sols['Pi'][0] = npy.zeros(Pi_i_in.shape) # Do not need Pi_f

        self.metric_a['a'] = 1 / (1 + z_i_in)
        self.metric_f['a'][-1] = 1 / (1 + z_i_in)

        self.Hubble_0 = Params['h'] * 100
        self.Omega_m = Params['Omega_m']
        self.Omega_L = Params['Omega_L']

        if( abs(self.Omega_m + self.Omega_L - 1.0) > 0.05 ):
            print("Warning! the total energy fraction is deviating from 1!!!")

        self.init()

        for step in  range(self.Ntau , -1, -1):
            self.sols['Phi'][step], self.sols['Pi'][step] = \
                self.Phi_Pi_gen(delta[step ], vw[step ], \
                                -(vw[step  + 1] - vw[step - 1]) \
                                /  (2 * (self.tau_i - self.tau_f)/self.Ntau), \
                                self.metric_f['Hubble'][step], self.metric_f['a'][step], self.to_tau(step))
            #self.sols['Phi'][step] -= self.sols['Phi'][step].mean()

        
        self.build_hier(alm_form = True)

        self.update_Pi(0)
        
        if(self.grid == 'DH'):
            self.grid = 'DH2'                

    
    def init_from_slice(self, z_i_in, r_max_in, delta_in, vw_in, Phi_i_in, Pi_i_in, Params, \
                        z_f_in, r_min_in, Phi_f_in):
        """Sample function of setting-up initial data
        Parameters
        ----------
        z_i_in : float
            initial z
        r_max_in: float
            initial radial distance at z
        delta_in: float, shape(N+2, NPIX) 
        delta_in: float, shape(N+2, NPIX) 
        Phi_i_in: float, shape(NPIX)
        Pi_i_in: float, shape(NPIX)
        Paras: list
            Cosmological parameters, needs to include 'h', 'Omega_m' and 'Omega_L' term
        z_f_in: float
            Final z
        r_min_in: float
            Final radial distance at z_f
        Phi_f_in: float, shape(NPIX)
        Returns
        -------
        void
        """
        self.r_max = r_max_in;

        self.tau_i = r_max_in;

        self.a_f = 1 / (1 + z_f_in)

        self.tau_f = r_min_in

        self.Ntau = delta_in.shape[0] - 2
        

        for field in self.metric_f:
            self.metric_f[field] = npy.zeros(self.Ntau + 1)

        #descritation index ranging from 0 to Ntau
        if(self.grid == 'GLQ' or self.grid == 'DH' or self.grid == 'DH2'):    
            for field in self.sols:
                self.sols[field] = npy.zeros((self.Ntau + 1, self.nlat, self.nlon))
        else:
            for field in self.sols:
                self.sols[field] = npy.zeros((self.Ntau + 1, self.Npix))
            

        self.matter['delta'] = delta_in[0:self.Ntau+1].copy()
        self.matter['vw'] = vw_in[0:self.Ntau+1].copy()
        # Need to substract extra 2 padding grid 

        
        self.sols['Phi'][-1] = Phi_i_in.copy()
        self.sols['Pi'][-1] = Pi_i_in.copy()


        self.sols['Phi'][0] = Phi_f_in.copy()
        #self.sols['Pi'][0] = Pi_f_in.copy()
        self.sols['Pi'][0] = npy.zeros(Pi_i_in.shape) # Do not need Pi_f

        self.metric_a['a'] = 1 / (1 + z_i_in)
        self.metric_f['a'][-1] = 1 / (1 + z_i_in)

        self.Hubble_0 = Params['h'] * 100
        self.Omega_m = Params['Omega_m']
        self.Omega_L = Params['Omega_L']

        if( abs(self.Omega_m + self.Omega_L - 1.0) > 0.05 ):
            print("Warning! the total energy fraction is deviating from 1!!!")

        self.init()

        
        self.build_hier()

        self.update_Pi(0)
        
        if(self.grid == 'DH'):
            self.grid = 'DH2'                
            
        
    def build_lcmetric(self):
        self.MG()
        #self.sols['Phi'][1:self.Ntau-1] = self.to_real(self.Phi_hier[0][1:self.Ntau-1])
        #self.sols['Pi'][1:self.Ntau-1] = self.to_real(self.Pi_hier[0][1:self.Ntau-1])

    def dPi_dr(self):
        Omega = ut.np_fderv1(self.Phi_hier[0], -dtau_hier[0], 0)
        Omega_dot = ut.np_fderv2(self.Phi_hier[0], -dtau_hier[0], 0)
        Pi_dot = ut.np_fderv1(self.Pi_hier[0], -dtau_hier[0], 0)
        dPi_dr = -2 * Pi_dot - Omega_dot - 3 * self.Hubble_hier[0][:,None] * \
            (Omega + self.Pi_hier[0]) \
            - (2 * self.Hubble_dt_hier[0][:,None] + self.Hubble_hier[0][:,None]**2) * self.Phi_hier[0]
        return to_real(dPi_dr, self.lmax, array_data = False)
        
        
    #------------------Starting set-up integration to correcction terms-----------#


    Da_corrs = {'Da_S':None, 'Da_v':None, 'Da_SW':None, 'DA_ISW':None}

    z_corrs = {'z_S':None, 'z_v':None, 'z_SW':None, 'z_ISW':None}

    def init_corrs(self):
        for corrs in Da_corrs:
            Da_corrs[corrs] = npy.zeros(self.Phi_hier[0].shape)

        for corrs in z_corrs:
            z_corrs[corrs] = npy.zeros(self.Phi_hier[0].shape)

    def dDa_S_dt(self, step):
        return 0
            
    def cal_corrs(self):
        return 0
    












