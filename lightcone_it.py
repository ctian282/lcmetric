import numpy as npy
import healpy as hp

import lcmetric.utils

class Lightcone:

    # does not really use rho_bar, replaced by H
    matter={'rho_bar':None, 'delta':None, 'vw':None}


    # For iteration scheme
    metric_a={ 'a':None, }
    metric_dt = {'a':None, }
    metric_f = {'a':None, 'Hubble':None, 'Hubble_dt':None}

    sols = {'Phi':None, 'Pi':None, 'Omega':None}

    Hubble_0 = None
    Hubble = None
    Hubble_dt = None

    angle_lap = None

    nside = None

    theta_list = None

    phi_list = None
    
    def  __init__(self, delta_in, vw_in, nside_in, lmax_in = None, \
                  epsilon_in = 1e-6, tol_in= 1e-7, niter_in = 100):

        if lmax_in is None: lmax_in = 2*nside_in - 1
        
        self.matter['delta'] = delta_in.copy()
        self.matter['vw'] = vw_in.copy()
        # Need to substract extra 2 padding grid 
        self.Ntau = delta_in.shape[0] - 2
        self.nside = nside_in
        self.Npix = 12 * self.nside**2
        
        self.theta_list, self.phi_list = hp.pix2ang(self.nside, range(12*self.nside**2))

        self.epsilon = epsilon_in
        self.tol = tol_in
        self.niter = niter_in
        
        # descritation index ranging from 0 to Ntau
        for field in self.metric_f:
            self.metric_f[field] = npy.zeros(self.Ntau + 1)

        for field in self.sols:
            self.sols[field] = npy.zeros((self.Ntau + 1, 12*self.nside**2))

        self.lmax = lmax_in
        self.lm = hp.Alm.getlm(lmax_in, npy.array(range( int(1+self.lmax + 0.5 * self.lmax*(1+self.lmax)) )))

            
    def to_tau(self, ntau):
        return (ntau) / self.Ntau * self.tau_i

    # ---------Functions of time derivatives-----------------------
        
    def da_dt(self, ntau):
        return self.metric_a['a'] * self.Hubble

    def dPhi_dt(self, ntau):
        return self.sols['Omega'][ntau]
    
    def dPi_dt(self, ntau):
        return (2 / self.to_tau(ntau) - 4*self.metric_f['Hubble'][ntau]) * self.sols['Pi'][ntau] \
            - 3 * self.metric_f['Hubble'][ntau] * self.sols['Omega'][ntau] \
            - 3 * self.metric_f['Hubble'][ntau]**2 * self.sols['Phi'][ntau] \
            + self.angle_lap \
            - 1.5 * self.Hubble_0**2 * self.Omega_m * self.matter['delta'][ntau] \
            / self.metric_f['a'][ntau]\
            - 1.5 * self.Hubble_0**2 * self.Omega_m * self.matter['vw'][ntau]

    def dOmega_dt(self, ntau):
        return (2 / self.to_tau(ntau) + 2*self.metric_f['Hubble'][ntau]) * self.sols['Pi'][ntau] \
            - (2 * self.metric_f['Hubble_dt'][ntau] -2 * self.metric_f['Hubble'][ntau]**2) \
            * self.sols['Phi'][ntau] \
            + 1.5 * self.Hubble_0**2 * self.Omega_m * self.matter['delta'][ntau] \
            / self.metric_f['a'][ntau] \
            - self.angle_lap + 3 * self.Hubble_0**2 * self.Omega_m * self.matter['vw'][ntau]
        
    
    # --------------------updating other fields-----------
    def update_other_field(self, ntau):
        it = 1000
        self.Hubble = self.Hubble_0 * self.metric_a['a']\
            * npy.sqrt(self.Omega_m * self.metric_a['a']**(-3) + self.Omega_L )
        self.Hubble_dt = self.Hubble_0**2 * self.metric_a['a'] * self.Omega_L \
            - self.Hubble_0**2 * self.Omega_m / (2*self.metric_a['a'])

        self.metric_f['Hubble'][ntau] = self.Hubble
        self.metric_f['Hubble_dt'][ntau] = self.Hubble_dt
        
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
        

    def time_advance(self):
        for step in reversed(range(1, self.Ntau + 1)):
            if(step % 10 == 0): print(step)
            self.rk2_time_advance_step(step, self.tau_i / self.Ntau)
        

    def est_errors(self):
        errs = {'Phi':0.0, 'Pi':0.0, 'Omega':0.0}
            
        for field in self.sols:
            for step in reversed(range(self.ntau_f + 1, self.Ntau)):
                self.est_angle_lap(step)
                errs[field] = npy.max([ \
                    npy.abs( (self.sols[field][step + 1] - self.sols[field][step - 1]) \
                             / (2.0 * self.tau_i / self.Ntau)\
                             - eval('self.d'+field+'_dt')(step)).max(), errs[field]])

        for field in self.sols:
            print('Max error for field '+field+' is '+str(errs[field]))

    def est_angle_lap(self, ntau):
        it = 100
        alm = hp.map2alm(self.sols['Phi'][ntau], lmax = self.lmax, iter = it)
        alm *= -self.lm[0] *(self.lm[0] + 1)
        self.angle_lap = hp.alm2map(alm, self.nside, pixwin=False) / self.to_tau(ntau)**2
        self.angle_alp = 0
            
                
    def iteration(self):
        nit = self.niter
        while(nit > 0):
            if(nit % 10 == 0):
                print('For iteration '+str(nit))
                self.est_errors()
            nit -= 1
            for field in self.sols:
                for step in  reversed(range(self.ntau_f + 1, self.Ntau)):
                    self.est_angle_lap(step)
                    self.sols[field][step] -= \
                        ((self.sols[field][step + 1] - self.sols[field][step - 1]) \
                         / (2.0 * self.tau_i / self.Ntau) \
                         - eval('self.d'+field+'_dt')(step)) * self.epsilon
        
        
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
        self.sols['Omega'][self.Ntau] = Omega_i_in.copy()


        self.sols['Phi'][self.ntau_f] = Phi_f_in.copy()
        self.sols['Pi'][self.ntau_f] = Pi_f_in.copy()
        self.sols['Omega'][self.ntau_f] = Omega_f_in.copy()

        
        self.metric_a['a'] = 1 / (1 + z_i_in)
        self.metric_f['a'][self.Ntau] = 1 / (1 + z_i_in)
        
        self.Hubble_0 = Params['h'] * 100
        self.Omega_m = Params['Omega_m']
        self.Omega_L = Params['Omega_L']

        if( abs(self.Omega_m + self.Omega_L - 1.0) > 0.05 ):
            print("Warning! the total energy fraction is deviating from 1!!!")

        self.scheme = 'iteration'

        self.init()
        
    def build_lcmetric(self):
        self.iteration()
