import numpy as npy
import healpy as hp
#import numexpr as ne


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
    
    def  __init__(self, delta_in, vw_in, nlat_in, lmax_in = None, \
                  epsilon_in = 1e-6, tol_in= 1e-7, niter_in = 100):

        self.matter['delta'] = delta_in.copy()
        self.matter['vw'] = vw_in.copy()
        # Need to substract extra 2 padding grid 
        self.Ntau = delta_in.shape[0] - 2
        self.nlat = nlat_in
        self.nlon = self.nlat * 2
        self.Npix = self.nlat * self.nlon
        self.lmax = lmax_in

        self.epsilon = epsilon_in
        self.tol = tol_in
        self.niter = niter_in
        
        # descritation index ranging from 0 to Ntau
        for field in self.metric_f:
            self.metric_f[field] = npy.zeros(self.Ntau + 1)

        for field in self.sols:
            self.sols[field] = npy.zeros((self.Ntau + 1, self.nlat, self.nlon))

        if(lmax_in == None ): self.lmax = nlat_in - 1
        self.lm = npy.array([[ [-l*(l+1) for m in range(self.lmax+1)] for l in range(self.lmax+1)],
                             [ [-l*(l+1) for m in range(self.lmax+1)] for l in range(self.lmax+1)]])
            
    def to_tau(self, ntau):
        return (ntau) / self.Ntau * self.tau_i

    # ---------Functions of time derivatives-----------------------
        
    def da_dt(self, ntau):
        #return 0
        return self.metric_a['a'] * self.Hubble_a

    def dPhi_dt(self, ntau):
        return (-2 / self.tau_list[ntau] + 2*self.Hubble[ntau]) * self.Pi[ntau] \
            - (2 * self.Hubble_dt[ntau] -2 * self.Hubble[ntau]**2) \
            * self.Phi[ntau] \
            + 1.5 * self.Hubble_0**2 * self.Omega_m / self.a[ntau] * self.delta[ntau] \
            - self.angle_lap + 3 * self.Hubble_0**2 * self.Omega_m * self.vw[ntau]

        # return ne.evaluate("(-2 / tau_list + 2*Hubble) * Pi \
        #     - (2 * Hubble_dt -2 * Hubble**2) \
        #     * Phi \
        #     + 1.5 * Hubble_0**2 * Omega_m / a * delta \
        #     - angle_lap + 3 * Hubble_0**2 * Omega_m * vw",   \
        # local_dict = {'tau_list': self.tau_list[ntau], 'Hubble': self.Hubble[ntau], \
        #               'Hubble_dt':self.Hubble_dt[ntau],\
        #               'Pi':self.Pi[ntau], 'Phi':self.Phi[ntau], \
        #               'Hubble_0': self.Hubble_0, 'Omega_m':self.Omega_m,\
        #               'a':self.a[ntau], 'delta':self.delta[ntau], 'angle_lap':self.angle_lap, \
        #               'vw':self.vw[ntau]})

    
    def dPi_dt(self, ntau):
        if (ntau == self.Ntau):
            Omega_dot = (self.Phi[ntau] + self.Phi[ntau-2] - 2*self.Phi[ntau-1]) \
                / (self.tau_i / self.Ntau)**2
            Omega = -(self.Phi[ntau] - self.Phi[ntau-1]) / (self.tau_i / self.Ntau)
        else:
            Omega_dot = (self.Phi[ntau+1] + self.Phi[ntau-1] - 2*self.Phi[ntau]) \
                / (self.tau_i / self.Ntau)**2
            Omega = -(self.Phi[ntau + 1] - self.Phi[ntau-1]) / (2.0 * self.tau_i / self.Ntau)

        return - Omega_dot - 3 * self.Hubble[ntau] * Omega \
            - 2 * self.Hubble[ntau] * self.Pi[ntau] \
            - (2 * self.Hubble_dt[ntau] + self.Hubble[ntau]**2) * self.Phi[ntau] \
            + 1.5 * self.Hubble_0**2 * self.Omega_m * self.vw[ntau]
                       
    
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

        self.update_other_field(ntau)

        if(ntau == 1):
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


    def est_errors(self):
        errs = {'Phi':0.0, 'Pi':0.0, 'Omega':0.0}
        rel_err = 0
        mag = 0
        for field in self.sols:
            for step in reversed(range(self.ntau_f + 1, self.Ntau)):
                self.est_angle_lap(step)
                if field == 'Phi':
                    rhs = eval('self.d'+field+'_dt')(step)
                    lhs = (self.sols[field][step + 1] + self.sols[field][step - 1] \
                           - 2.0 * self.sols[field][step ])/ (self.tau_i / self.Ntau)**2

                    errs[field] = npy.max([npy.abs( lhs - rhs).max(), errs[field]])
                    rel_err += npy.linalg.norm(lhs - rhs)**2
                    mag += npy.linalg.norm(lhs)**2
                # elif field == 'Pi':
                #     errs[field] = npy.max([ \
                #                 npy.abs( -(self.sols[field][step + 1] - self.sols[field][step - 1]) \
                #                          / (2.0 * self.tau_i / self.Ntau)\
                #                          - eval('self.d'+field+'_dt')(step)).max(), errs[field]])
        rel_err = npy.sqrt(rel_err / mag)
        for field in self.sols:
            if field == 'Phi':
                print('Max error for field '+field+' is '+str(errs[field]))
                print('Relative error of the L2 norm is ' + str(rel_err))

    def est_angle_lap(self, ntau):
        alm=self.sh_grid[ntau].expand(lmax_calc=self.lmax)
        alm.coeffs*=self.lm
        self.angle_lap = alm.expand(grid='GLQ').data / self.tau_list[ntau]**2
        #Self.angle_lap = 0
            
    def iteration(self):
        nit = self.niter
        while(nit > 0):
            if(nit % 1 == 0):
                print('For iteration '+str(nit))
                self.est_errors()
            nit -= 1
            #raise ValueError('A very specific bad thing happened.')
            for field in self.sols:
                #for step in  reversed(range(self.ntau_f + 1, self.Ntau)):
                for step in  range(self.Ntau-1, self.ntau_f, -1):
                    self.est_angle_lap(step)
                    if field == 'Phi':
                        self.Phi[step] = (self.Phi[step + 1] + self.Phi[step - 1] \
                                     - (self.tau_i / self.Ntau)**2 * eval('self.d'+field+'_dt')(step)) / 2
                    elif field == 'Pi':
                        dt = eval('self.d'+field+'_dt')(step + 1)
                        self.Pi[step] = self.Pi[step+1] \
                            + dt * (self.tau_i / self.Ntau)
                        dt += eval('self.d'+field+'_dt')(step)
                        self.Pi[step] = self.Pi[step+1] \
                            + 0.5 * self.tau_i / self.Ntau * dt
        
        
    def init(self):
        #set-up time evolution of a and Hubble and Hubble_dt
        self.update_other_field(self.Ntau)
        for step in reversed(range(1, self.Ntau + 1)):
            self.rk2_time_advance_step(step, self.tau_i / self.Ntau)

        #set-up the initial guess
        for step in  reversed(range(self.ntau_f + 1, self.Ntau)):
            for field in self.sols:
                self.sols[field][step] = self.sols[field][self.Ntau] \
                    + (self.Ntau - step) / (self.Ntau - self.ntau_f) \
                    * (self.sols[field][self.ntau_f] - self.sols[field][self.Ntau])
            
    #-------------Initializations------------------------------------
    def init_from_slice(self, z_i_in, r_max_in, Phi_i_in, Pi_i_in, Omega_i_in, Params, \
                        z_f_in, r_min_in, Phi_f_in, Pi_f_in, Omega_f_in):
        self.r_max = r_max_in;

        self.tau_i = r_max_in;

        self.a_f = 1 / (1 + z_f_in)

        self.tau_f = r_min_in

        self.ntau_f = int(npy.round(self.tau_f / (self.tau_i / self.Ntau)))

        self.sols['Phi'][self.Ntau] = Phi_i_in.copy()
        self.sols['Pi'][self.Ntau] = Pi_i_in.copy()


        self.sols['Phi'][self.ntau_f] = Phi_f_in.copy()
        self.sols['Pi'][self.ntau_f] = Pi_f_in.copy()

        self.sh_grid = \
            npy.array([pysh.SHGrid.from_array(self.sols['Phi'][n], grid = 'GLQ', copy = False) \
                       for n in range(self.Ntau+1)])
        self.Phi = self.sols['Phi']
        self.Pi = self.sols['Pi']
        self.delta = self.matter['delta']
        self.vw = self.matter['vw']
        self.a = self.metric_f['a']
        self.Hubble = self.metric_f['Hubble']
        self.Hubble_dt = self.metric_f['Hubble_dt']

        self.tau_list = npy.array([self.to_tau(i) for i in range(self.Ntau+1)])
        
        self.metric_a['a'] = 1 / (1 + z_i_in)
        self.metric_f['a'][self.Ntau] = 1 / (1 + z_i_in)

        # self.metric_a['a'] = 1
        # self.metric_f['a'][self.Ntau] = 1

        
        self.Hubble_0 = Params['h'] * 100
        self.Omega_m = Params['Omega_m']
        self.Omega_L = Params['Omega_L']

        if( abs(self.Omega_m + self.Omega_L - 1.0) > 0.05 ):
            print("Warning! the total energy fraction is deviating from 1!!!")

        self.scheme = 'iteration'

        self.init()
        
    def build_lcmetric(self):
        self.iteration()
