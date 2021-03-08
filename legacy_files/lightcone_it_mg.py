
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

    angle_lap = None

    nside = None

    theta_list = None

    phi_list = None
    
    def  __init__(self,  nlat_in, lmax_in = None, \
                  epsilon_in = 1e-6, tol_in= 1e-7, niter_in = 100, use_SOR = False, grid = 'GLQ',
                  angle_lap_type = 'SPH', depth = 3, n_vcycles = 10, npre = 10, npost = 10):

        self.nlat = nlat_in
        self.nlon = self.nlat * 2
        self.Npix = self.nlat * self.nlon
        self.lmax = lmax_in
        self.use_SOR = use_SOR
        self.grid = grid
        self.angle_lap_type = angle_lap_type

        self.depth = depth
        self.n_vcycles = n_vcycles
        self.npre = npre
        self.npost = npost
        
        if(grid == 'GLQ'): self.extend = True
        else: self.extend = False
        
        self.epsilon = epsilon_in
        self.tol = tol_in
        self.niter = niter_in
        


    def to_tau(self, ntau, d = 0):
        return (ntau )* (2**d * self.tau_i ) / ( (self.Ntau) ) + self.ntau_f * self.tau_i / self.Ntau 

    # ---------Functions of time derivatives-----------------------
        
    def da_dt(self, ntau):
        #return 0
        return self.metric_a['a'] * self.Hubble_a

    def dPhi_dt(self, ntau, d):
        return (-2 / self.tau_hier[d][ntau] + 2*self.Hubble_hier[d][ntau]) * self.Pi_hier[d][ntau] \
            - (2 * self.Hubble_dt_hier[d][ntau] -2 * self.Hubble_hier[d][ntau]**2) \
            * self.Phi_hier[d][ntau] \
            + 1.5 * self.Hubble_0**2 * self.Omega_m / self.a_hier[d][ntau] * self.delta_hier[d][ntau] \
            - self.angle_lap + 3 * self.Hubble_0**2 * self.Omega_m * self.vw_hier[d][ntau]

    
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
            + 1.5 * self.Hubble_0**2 * self.Omega_m * self.vw_hier[d][ntau]
                       
    
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
        #----------------first phase-----------------
        for field in self.metric_a:
            self.metric_dt[field] = eval('self.d'+field+'_dt')(ntau)

        for field in self.metric_a:
            self.metric_a[field] += self.metric_dt[field] * dtau
            self.metric_f[field][ntau-1] = self.metric_f[field][ntau] + 0.5 * dtau * self.metric_dt[field] 

        self.update_other_field(ntau-1)

        if(ntau == -self.ntau_f + 1):
            for field in self.metric_a:
                self.metric_f[field][0] = self.metric_a[field].copy()
            return
        
        #------------------second phase--------------
        ntau = ntau - 1

        for field in self.metric_a:
            self.metric_dt[field] = eval('self.d'+field+'_dt')(ntau)


        for field in self.metric_a:
            self.metric_f[field][ntau] += 0.5 * dtau * self.metric_dt[field] 
            self.metric_a[field] = self.metric_f[field][ntau].copy()

        self.update_other_field(ntau)


    def est_errors(self, d, indent = ''):
        err = 0
        rel_err = 0
        mag = 0
        max_step = -1
        for step in reversed(range(1, self.h_size[d]-1)):
            self.est_angle_lap(step, d)
            rhs = self.dPhi_dt(step, d) + self.rhs_hier[d][step]
            lhs = (self.Phi_hier[d][step + 1] + self.Phi_hier[d][step - 1] \
                   - 2.0 * self.Phi_hier[d][step ])/ self.dtau_hier[d]**2 
            if(self.grid=='DH2'):
                bak = err
                err = npy.max([npy.abs( lhs - rhs)[1:,:].max(), err])
                if(err > bak): max_step = step
                rel_err += npy.linalg.norm( (lhs - rhs)[1:,:] )**2
                mag += npy.linalg.norm(lhs[1:,:])**2
            else:
                err = npy.max([npy.abs( lhs - rhs).max(), err])
                rel_err += npy.linalg.norm( (lhs - rhs) )**2
                mag += npy.linalg.norm(lhs)**2                    
        rel_err = npy.sqrt(rel_err / mag)
        for field in self.sols:
            if field == 'Phi':
                print(indent+'Max error for field '+field+' is '+str(err) + ' at step '+str(max_step))
                print(indent+'Relative error of the L2 norm is ' + str(rel_err))
                        
    def est_angle_lap(self, ntau, d):
        alm=self.sh_grid_hier[d][ntau].expand(lmax_calc=self.lmax)
        alm.coeffs*=self.lm
        self.angle_lap = alm.expand(grid=self.grid, extend = self.extend).data / self.tau_hier[d][ntau]**2
            
    def relax(self, d, nsteps):
        err_norm = 1e100
        while(nsteps > 0):
            nsteps -= 1
            #raise ValueError('A very specific bad thing happened.')
            # last_Phi = self.Phi_hier[d].copy()
            # last_Pi = self.Pi_hier[d].copy()
            for field in self.sols:
                #for step in  reversed(range(self.ntau_f + 1, self.Ntau)):
                for step in  range(self.h_size[d] - 2, 0, -1):
                    if field == 'Phi':
                        self.est_angle_lap(step, d)
                        self.Phi_hier[d][step] = (self.Phi_hier[d][step + 1] + self.Phi_hier[d][step - 1] - \
                                self.dtau_hier[d]**2 \
                                * (eval('self.d'+field+'_dt')(step, d) + self.rhs_hier[d][step]) ) / 2
                    elif field == 'Pi':
                        dt = eval('self.d'+field+'_dt')(step + 1, d)
                        self.Pi_hier[d][step] = self.Pi_hier[d][step+1] \
                            + dt * self.dtau_hier[d]
                        dt += eval('self.d'+field+'_dt')(step, d)
                        self.Pi_hier[d][step] = self.Pi_hier[d][step+1] \
                            + 0.5 * self.dtau_hier[d] * dt
                        
            # temp_err = 0
            # for step in range(self.h_size[d] - 2, 0, -1):
            #     rhs = self.dPhi_dt(step, d) + self.rhs_hier[d][step]
            #     lhs = (self.Phi_hier[d][step + 1] + self.Phi_hier[d][step - 1] \
            #            - 2.0 * self.Phi_hier[d][step ])/ self.dtau_hier[d]**2
            #     temp_err += npy.linalg.norm( (lhs - rhs)[1:,:] )**2
            # if(temp_err > err_norm):
            #     self.Phi_hier[d] = last_Phi
            #     self.Pi_hier[d] = last_Pi
            #     return
            # err_norm = temp_err


            
    def generate_rl(self, d):
        for step in  range(self.h_size[d] - 2, 0, -1):
            self.est_angle_lap(step, d)
            L =\
                (self.Phi_hier[d][step + 1] + self.Phi_hier[d][step - 1] \
                 - 2.0 * self.Phi_hier[d][step ])/ (self.dtau_hier[d])**2
            self.rl_hier[d][step] = L - self.dPhi_dt(step, d)

        #self.rl_hier[d][0] = 
        
    def update_Pi(self, d):
        for step in  range(self.h_size[d] - 2, 0, -1):
            self.est_angle_lap(step, d)
            dt = self.dPi_dt(step + 1, d)
            self.Pi_hier[d][step] = self.Pi_hier[d][step+1] \
                + dt * self.dtau_hier[d]
            dt += self.dPi_dt(step, d)
            self.Pi_hier[d][step] = self.Pi_hier[d][step+1] \
                + 0.5 * self.dtau_hier[d] * dt

    def update_rhs(self, d):
        for step in  range(self.h_size[d] - 2, 0, -1):
            self.est_angle_lap(step, d)
            self.rhs_hier[d][step] += (self.Phi_hier[d][step + 1] + self.Phi_hier[d][step - 1] \
                                 - 2.0 * self.Phi_hier[d][step ])/ (self.dtau_hier[d])**2\
                                 - self.dPhi_dt(step, d) - self.rl_hier[d][step]

    def clear_field(self, data):
        for d in range(self.depth):
            data[d].fill(0) 

    def smoothing_Phi(self, d):
        for step in range(1, self.h_size[d]-1):
            alm=self.sh_grid_hier[d][step].expand(lmax_calc=self.lmax)
            self.Phi_hier[d][step] = alm.expand(grid=self.grid, extend = self.extend).data

        
            
    def FAS(self):
        n_vcycles = self.n_vcycles
        while(n_vcycles >0):
            n_vcycles -= 1
            print('Starting v-cycle '+ str(self.n_vcycles - n_vcycles))
            self.est_errors(0)
            #self.hier_restrict(self.Pi_hier,0)
            for d in range(0,self.depth-1):
                self.relax(d, int(self.npre/2**d ))
                print('    At level '+str(d)+', after '+str( int(self.npre/2**d) )+' relaxation')
                self.est_errors(d, '    ')
                if(d < self.depth - 1):
                    self.generate_rl(d)

                    self.hier_restrict(self.Pi_hier,d)
                    
                    self.hier_restrict(self.rl_hier,d)
                    self.hier_restrict(self.Phi_hier,d)
                    #self.smoothing_Phi(d+1)
                    self.update_Pi(d+1)
                    self.hier_restrict(self.rhs_hier, d)
                    self.update_rhs(d+1)
            self.clear_field(self.rl_hier)
            for d in range(self.depth - 1, 0, -1):
                self.relax(d, int(self.npost / 2**d) )
                print('    At level '+str(d)+', after '+str( int(self.npost / 2**d) )+' relaxation')
                self.est_errors(d, '    ') 
                if(d > 0):
                    self.hier_restrict(self.Phi_hier, d-1, self.rl_hier)
                    self.rl_hier[d] = - self.rl_hier[d] + self.Phi_hier[d]
                    self.hier_prolong(self.rl_hier, d)
                    self.Phi_hier[d-1] += self.rl_hier[d-1]
                    self.update_Pi(d-1)
            self.clear_field(self.rhs_hier)
            self.clear_field(self.rl_hier)
        
    def init(self):
        #set-up time evolution of a and Hubble and Hubble_dt
        self.update_other_field(self.Ntau-self.ntau_f)
        for step in reversed(range(1, self.Ntau-self.ntau_f + 1)):
            self.rk2_time_advance_step(step, self.tau_i / self.Ntau)

        #set-up the initial guess
        for step in  reversed(range(1, self.Ntau-self.ntau_f)):
            for field in self.sols:
                if(field == 'Phi'):
                    self.sols[field][step] = self.sols[field][self.Ntau-self.ntau_f] \
                        + (self.Ntau - self.ntau_f - step) / (self.Ntau - self.ntau_f) \
                        * (self.sols[field][0] - self.sols[field][self.Ntau - self.ntau_f])

                    
        #self.update_Pi(0)

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
        
    def build_hier(self):

        # Initializing finest hiers
        self.Phi_hier = [None] * self.depth
        self.Pi_hier = [None] * self.depth
        self.SHGrid_hier = [None] * self.depth
        self.delta_hier = [None] * self.depth
        self.vw_hier = [None] * self.depth
        self.a_hier = [None] * self.depth
        self.Hubble_hier = [None] * self.depth
        self.Hubble_dt_hier = [None] * self.depth
        self.lm_hier = [None] * self.depth
        self.tau_hier = [None] * self.depth
        self.dtau_hier = [None] * self.depth
        self.h_size = [None] * self.depth
        self.sh_grid_hier = [None] * self.depth

        self.rhs_hier = [None] * self.depth
        self.rl_hier = [None] * self.depth
        
        self.Phi_hier[0] = self.sols['Phi']
        self.Pi_hier[0] = self.sols['Pi']
        self.delta_hier[0] = self.matter['delta']
        self.vw_hier[0] = self.matter['vw']

        self.a_hier[0] = self.metric_f['a']
        self.Hubble_hier[0] = self.metric_f['Hubble']
        self.Hubble_dt_hier[0] = self.metric_f['Hubble_dt']

        self.rhs_hier[0] = npy.zeros((self.Ntau - self.ntau_f + 1, self.nlat, self.nlon))
        #self.err_hier[0] = npy.zeros((self.Ntau - self.ntau_f + 1, self.nlat, self.nlon))
        self.rl_hier[0] = npy.zeros((self.Ntau - self.ntau_f + 1, self.nlat, self.nlon))
        
        # Allocating space for hiers
        # and initializing data through hier_restrict
        self.h_size[0] = self.Ntau - self.ntau_f + 1
        n = self.h_size[0] #h_size includes both ends
        self.h_size[0] = n
        for d in range(self.depth-1):
            if(n %2 == 0):
                n = int(n /2)
            else:
                n = int((n + 1) / 2)
                
            self.h_size[d+1] = n
                
            self.Phi_hier[d+1] = npy.zeros((n, self.nlat, self.nlon))
            self.hier_restrict(self.Phi_hier, d)

            self.Pi_hier[d+1] = npy.zeros((n, self.nlat, self.nlon))
            
            self.delta_hier[d+1] = npy.zeros((n, self.nlat, self.nlon))
            self.hier_restrict(self.delta_hier, d)
            
            self.vw_hier[d+1] = npy.zeros((n, self.nlat, self.nlon))
            self.hier_restrict(self.vw_hier, d)

            
            self.rhs_hier[d+1] = npy.zeros((n, self.nlat, self.nlon))
            #self.err_hier[d+1] = npy.zeros((n, self.nlat, self.nlon))
            self.rl_hier[d+1] = npy.zeros((n, self.nlat, self.nlon))
                

            self.a_hier[d+1] = npy.zeros(n)
            self.hier_restrict(self.a_hier, d)
            
            self.Hubble_hier[d+1] = npy.zeros(n)
            self.hier_restrict(self.Hubble_hier, d)
            
            self.Hubble_dt_hier[d+1] = npy.zeros(n)
            self.hier_restrict(self.Hubble_dt_hier, d)
            

        # Initializing pre-caclulated lm and radius
        for d in range(self.depth):
            self.sh_grid_hier[d] = npy.array( \
                [pysh.SHGrid.from_array(self.Phi_hier[d][n], grid = self.grid, copy = False) \
                       for n in range(self.Phi_hier[d].shape[0])])

            self.tau_hier[d] = npy.array([self.to_tau(i, d) for i in range(self.h_size[d])])
            self.dtau_hier[d] = self.tau_i / self.Ntau * 2**d

        lmax = self.sh_grid_hier[0][0].lmax 
        self.lm = npy.array([[ [-l*(l+1) for m in range(lmax+1)] for l in range(lmax+1)],
                                  [ [-l*(l+1) for m in range(lmax+1)] for l in range(lmax+1)]])


        self.update_Pi(0)
        #for d in range(self.depth - 1):
            #self.hier_restrict(self.Pi_hier, d)

        
                
    #-------------Initializations------------------------------------
    def init_from_slice(self, z_i_in, r_max_in, delta_in, vw_in, Phi_i_in, Pi_i_in, Params, \
                        z_f_in, r_min_in, Phi_f_in, Pi_f_in):
        self.r_max = r_max_in;

        self.tau_i = r_max_in;

        self.a_f = 1 / (1 + z_f_in)

        self.tau_f = r_min_in

        self.Ntau = delta_in.shape[0] - 2
        
        self.ntau_f = int(npy.round(self.tau_f / (self.tau_i / self.Ntau)))

        #descritation index ranging from 0 to Ntau
        for field in self.metric_f:
            self.metric_f[field] = npy.zeros(self.Ntau - self.ntau_f + 1)

        for field in self.sols:
            self.sols[field] = npy.zeros((self.Ntau - self.ntau_f + 1, self.nlat, self.nlon))

        self.matter['delta'] = delta_in[self.ntau_f:self.Ntau+1].copy()
        self.matter['vw'] = vw_in[self.ntau_f:self.Ntau+1].copy()
        # Need to substract extra 2 padding grid 

        
        self.sols['Phi'][-1] = Phi_i_in.copy()
        self.sols['Pi'][-1] = Pi_i_in.copy()


        self.sols['Phi'][0] = Phi_f_in.copy()
        self.sols['Pi'][0] = Pi_f_in.copy()

        self.metric_a['a'] = 1 / (1 + z_i_in)
        self.metric_f['a'][-1] = 1 / (1 + z_i_in)

        # self.metric_a['a'] = 1
        # self.metric_f['a'][self.Ntau] = 1

        
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
            self.dtheta = npy.radians( npy.abs(self.sh_grid_hier[0][0].lats()[2] \
                                               - self.sh_grid_hier[0][0].lats()[1] ) )
            self.dphi = npy.radians(self.sh_grid_hier[0][0].lons()[2] - self.sh_grid_hier[0][0].lons()[1])

        self.theta_list = npy.radians(90 - self.sh_grid_hier[0][0].lats())
        self.phi_list = npy.radians(self.sh_grid_hier[0][0].lons())
                

        

        
        
        
    def build_lcmetric(self):
        self.FAS()











