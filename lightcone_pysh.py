import numpy as npy
import healpy as hp

import lcmetric.utils
import pyshtools as pysh

class Lightcone:

    # does not really use rho_bar, replaced by H
    matter={'rho_bar':None, 'delta':None, 'vw':None}


    # For 2nd order Runge Kutta
    metric_a={ 'a':None, 'Phi':None, 'Pi':None, 'Omega':None}
    metric_dt = {'a':None, 'Phi':None, 'Pi':None, 'Omega':None}
    metric_f = {'a':None, 'Phi':None, 'Pi':None, 'Omega':None}

    #homo_matter={'rho_bar':None}
    
    #homo_metric_a={'a':None}
    #homo_metric_dt={'a':None}
    #homo_metric_f={'a':None}

    
    Hubble_0 = None
    Hubble = None
    Hubble_dt = None

    angle_lap = None

    nside = None

    theta_list = None

    phi_list = None
    
    def  __init__(self, delta_in, vw_in, nlat_in, lmax_in=None):

        self.matter['delta'] = delta_in.copy()
        self.matter['vw'] = vw_in.copy()
        # Need to substract extra 2 padding grid 
        self.Ntau = delta_in.shape[0] - 2
        self.nlat = nlat_in
        self.nlon = self.nlat * 2
        self.Npix = self.nlat * self.nlon
        self.lmax = lmax_in

        # descritation index ranging from 0 to Ntau
        for field in self.metric_f:
            if(field == 'a'):
                self.metric_f[field] = npy.zeros(self.Ntau + 1)
                self.metric_dt[field] = npy.zeros(self.Ntau + 1)
                continue
            
            self.metric_f[field] = npy.zeros((self.Ntau + 1, self.nlat, self.nlon))
            self.metric_dt[field] = npy.zeros((self.Ntau + 1, self.nlat, self.nlon))

        if(lmax_in == None ): self.lmax = nlat_in - 1
        self.lm = npy.array([[ [-l*(l+1) for m in range(self.lmax+1)] for l in range(self.lmax+1)],
     [ [-l*(l+1) for m in range(self.lmax+1)] for l in range(self.lmax+1)]])

            
    def to_tau(self, ntau):
        return (ntau) / self.Ntau * self.tau_i

    # ---------Functions of time derivatives-----------------------
        
    def da_dt(self, ntau):
        #return 0
        return self.metric_a['a'] * self.Hubble

    def dPhi_dt(self, ntau):
        return self.metric_a['Omega']
    
    def dPi_dt(self, ntau):
        #return self.angle_lap
        return (2 / self.to_tau(ntau) - 4*self.Hubble) * self.metric_a['Pi'] \
            - 3 * self.Hubble * self.metric_a['Omega'] \
            - 3 * self.Hubble**2 * self.metric_a['Phi'] \
            + self.angle_lap \
            - 1.5 * self.Hubble_0**2 * self.Omega_m * self.matter['delta'][ntau] / self.metric_a['a']\
            - 1.5 * self.Hubble_0**2 * self.Omega_m * self.matter['vw'][ntau]
    
    def dOmega_dt(self, ntau):
        #return self.angle_lap
        return (-2 / self.to_tau(ntau) + 2*self.Hubble) * self.metric_a['Pi'] \
            - (2 * self.Hubble_dt -2 * self.Hubble**2) \
            * self.metric_a['Phi'] \
            + 1.5 * self.Hubble_0**2 * self.Omega_m * self.matter['delta'][ntau] \
            / self.metric_a['a'] \
            - self.angle_lap + 3 * self.Hubble_0**2 * self.Omega_m * self.matter['vw'][ntau]

    
    # --------------------updating other fields-----------
    def update_other_field(self, ntau):
        self.Hubble = self.Hubble_0 * self.metric_a['a']\
            * npy.sqrt(self.Omega_m * self.metric_a['a']**(-3) + self.Omega_L )
        self.Hubble_dt = self.Hubble_0**2 * self.metric_a['a']**2 * self.Omega_L \
            - self.Hubble_0**2 * self.Omega_m / (2*self.metric_a['a'])

        # self.Hubble = 0
        # self.Hubble_dt = 0

        alm=self.sh_grid.expand(lmax_calc=self.lmax)
        alm.coeffs*=self.lm
        self.angle_lap = alm.expand(grid='GLQ').data / self.to_tau(ntau)**2
        #self.angle_lap = 0
    
    #-----------Time advancing----------------------------------
    # RK2
    def rk2_time_advance_step(self, ntau, dtau):
        #----------------first phase-----------------
        for field in self.metric_a:
            self.metric_dt[field][ntau] = eval('self.d'+field+'_dt')(ntau)


        for field in self.metric_a:
            self.metric_a[field] += self.metric_dt[field][ntau] * dtau
            self.metric_f[field][ntau-1] = self.metric_f[field][ntau] + 0.5 * dtau * self.metric_dt[field][ntau] 

        self.update_other_field(ntau)

        if(ntau == 1):
            for field in self.metric_a:
                self.metric_f[field][0] = self.metric_a[field].copy()
            return
        
        #------------------second phase--------------
        ntau = ntau - 1

        for field in self.metric_a:
            self.metric_dt[field][ntau] = eval('self.d'+field+'_dt')(ntau)


        for field in self.metric_a:
            self.metric_f[field][ntau] += 0.5 * dtau * self.metric_dt[field][ntau] 
            self.metric_a[field] = self.metric_f[field][ntau].copy()

        self.update_other_field(ntau)

    # -----------------iterative Crank–Nicholson scheme---------------------
    # def cn_time_advance_step(self, ntau, dtau):
    #     #----------------first phase-----------------
    #     for field in self.metric_a:
    #         self.metric_dt[field] = eval('self.d'+field+'_dt')(ntau)


    #     for field in self.metric_a:
    #         self.metric_a[field] += self.metric_dt[field] * dtau
    #         #self.metric_f[field][ntau-1] = self.metric_f[field][ntau] + 0.5 * dtau * self.metric_dt[field] 

    #     self.update_other_field(ntau)

    #     if(ntau == 1):
    #         for field in self.metric_a:
    #             self.metric_f[field][0] = self.metric_a[field].copy()
    #         return

    #     #---------------second phase-----------------------------------
    #     for field in self.metric_a:
    #         self.metric_dt += eval('self.d'+field+'_dt')(ntau) 
    #         self.metric_dt *= 0.5

    #     for field in self.metric_a:
    #         self.metric_f[field][ntau - 1] += self.metric_dt[field] * dtau
    #         self.metric_a[field] = self.metric_f[field][natu-1].copy()

    #     self.update_other_field(ntau)
        

    def time_advance(self):
        for step in reversed(range(80, self.Ntau + 1)):
            if(step % 100 == 0): print(step)
            self.rk2_time_advance_step(step, -self.tau_i / self.Ntau)
        

    #-------------Initializations------------------------------------
    def init_from_slice(self, z_i_in, r_max_in, Phi_i_in, Pi_i_in, Omega_i_in, Params):
        self.r_max = r_max_in;

        self.tau_i = r_max_in;
        
        self.metric_a['Phi'] = Phi_i_in.copy();
        self.metric_f['Phi'][self.Ntau] = Phi_i_in.copy();

        self.metric_a['Pi'] = Pi_i_in.copy();
        self.metric_f['Pi'][self.Ntau] = Pi_i_in.copy();

        self.metric_a['Omega'] = Omega_i_in.copy();
        self.metric_f['Omega'][self.Ntau] = Omega_i_in.copy();

        self.sh_grid = \
            pysh.SHGrid.from_array(self.metric_a['Phi'], grid = 'GLQ', copy = False)

        

        self.metric_a['a'] = 1 / (1 + z_i_in)
        self.metric_f['a'][self.Ntau] = 1 / (1 + z_i_in)

        # self.metric_a['a'] = 1
        # self.metric_f['a'][self.Ntau] = 1

        self.Hubble_0 = Params['h'] * 100
        self.Omega_m = Params['Omega_m']
        self.Omega_L = Params['Omega_L']
        
        if( abs(self.Omega_m + self.Omega_L - 1.0) > 0.05 ):
            print("Warning! the total energy fraction is deviating from 1!!!")
        

    def build_lcmetric(self):
        self.update_other_field(self.Ntau)
        self.time_advance()
