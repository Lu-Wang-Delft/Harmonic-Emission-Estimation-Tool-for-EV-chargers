import tensorflow as tf
import numpy as np
from tensorflow.linalg import inv
import matplotlib.pyplot as plt
#***********************************************#
#------------------ Model 1 --------------------#
#***********************************************#
class Impedance_model1():  # The design of this model is: CC(SRF-PI)+PLL(SRF-PLL)+DVC(PI)+filter(L-)+modulation(SPWM)+topology(2L or Vienna)
    def __init__(self, Rfilter, Lfilter,kpi,kii, kppll, kipll, kpu,kiu,fsw,
                Cout,Vdc,f1,Vsdq,Power,f_sweep): # The input dtype is float32
        # change numpy value to tensorflow value        
        Rfilter_ts = tf.constant([Rfilter], dtype=tf.float32)
        Lfilter_ts = tf.constant([Lfilter], dtype=tf.float32)
        kpi_ts = tf.constant([kpi], dtype=tf.float32)
        kii_ts = tf.constant([kii], dtype=tf.float32)
        kppll_ts = tf.constant([kppll], dtype=tf.float32)
        kipll_ts = tf.constant([kipll], dtype=tf.float32)
        kpu_ts = tf.constant([kpu], dtype=tf.float32)
        kiu_ts = tf.constant([kiu], dtype=tf.float32)
        fsw_ts = tf.constant([fsw], dtype=tf.float32)
        Cout_ts = tf.constant([Cout], dtype=tf.float32)
        Vdc_ts = tf.constant([Vdc], dtype=tf.float32)
        f_sweep_ts = tf.constant(f_sweep,dtype=tf.int32)

        self.Rfilter = tf.cast(Rfilter_ts,dtype=tf.complex64)
        self.Lfilter = tf.cast(Lfilter_ts,dtype=tf.complex64)
        self.kpi = tf.cast(kpi_ts,dtype=tf.complex64)
        self.kii = tf.cast(kii_ts,dtype=tf.complex64)
        self.kppll = tf.cast(kppll_ts,dtype=tf.complex64)
        self.kipll = tf.cast(kipll_ts,dtype=tf.complex64)
        self.kpu = tf.cast(kpu_ts,dtype=tf.complex64)
        self.kiu = tf.cast(kiu_ts,dtype=tf.complex64)
        self.fsw = tf.cast(fsw_ts,dtype=tf.complex64)
        self.Cout = tf.cast(Cout_ts/2,dtype=tf.complex64)
        self.Vdc = tf.cast(Vdc_ts,dtype=tf.complex64)
        self.Vsd = Vsdq[0]
        self.Vsq = Vsdq[1]
        self.Pi = tf.constant(np.pi,dtype=tf.complex64)
        self.w1 = 2*self.Pi*f1
        self.P = Power

        # Calculate the delay of the control
        self.Tdel = 1.5/self.fsw
        # Calculate the s domain
        self.S_domain(f_sweep_ts)
        # Calculate the steady state 
        self.cal_ss()
        # print('Isd = {:f} and Isq = {:f}'.format(self.Isdq.numpy()[0,0].real, self.Isdq.numpy()[1,0].real))
        # print('Dcd = {:f} and Dcq = {:f}'.format(self.Dcdq.numpy()[0,0].real, self.Dcdq.numpy()[1,0].real))
        # print('Dsd = {:f} and Dsq = {:f}'.format(self.Dsdq.numpy()[0,0].real, self.Dsdq.numpy()[1,0].real))
        # Calculate all the blocks
        self.cal_blocks()
        # Calculate the impedance
        self.cal_impedance()
    
    def S_domain(self,f_sweep): # Define the S domain used for the impedance calculation
        self.f_ab = tf.cast(f_sweep,dtype=tf.complex64) # The input frequency is given in ab frame
        f_dq_n=tf.reverse(-f_sweep-50,[0])
        self.f_dq = tf.concat((f_dq_n,f_sweep-50),0)
        f_dq = tf.concat((-f_sweep-50,f_sweep-50),0) # This is the dq frame frequency used for ab-frame impedance calculation
        f_dq_cplx = tf.cast(f_dq, dtype=tf.complex64)
        self.s = 1j*2*self.Pi*f_dq_cplx; # s domain parameter
        self.s_3 = tf.expand_dims(tf.expand_dims(self.s,axis = 1),axis = 1) # change the dimension of self.s from (size,) to (size,1,1) for calculation
        
    
    def cal_ss(self): # Calculate the steady state
        self.Id_ref = 1/2/self.Rfilter * (self.Vsd - tf.sqrt(self.Vsd**2 - 8/3*self.P*self.Rfilter))
        self.Iq_ref = tf.constant([0],dtype=tf.complex64)
        self.Isdq = tf.cast(tf.concat([[tf.math.real(self.Id_ref)], [tf.math.real(self.Iq_ref)]],0),dtype=tf.complex64)
        Dcdq = 1/self.Vdc*(self.Vsd-self.Rfilter*self.Isdq[0]-1j*self.w1*self.Lfilter*self.Isdq[0])
        Dsdq = Dcdq
        self.Dcdq = tf.cast(tf.concat([[tf.math.real(Dcdq)], [tf.math.imag(Dcdq)]],0),dtype=tf.complex64)
        self.Dsdq = tf.cast(tf.concat([[tf.math.real(Dsdq)], [tf.math.imag(Dsdq)]],0),dtype=tf.complex64)
        self.Rl = self.Vdc**2/self.P
        self.IsdqT = tf.transpose(self.Isdq)
        self.DcdqT = tf.transpose(self.Dcdq)
        self.DsdqT = tf.transpose(self.Dsdq)
    

    def cal_blocks(self): # Calculate all the blocks
        # Define all the blocks first
        self.J = tf.constant([[0, -1],[1, 0]],dtype=tf.complex64)
        self.I = tf.constant([[1, 0],[0, 1]],dtype=tf.complex64)
        V_i = tf.constant([[1],[0]],dtype=tf.complex64)
        Tpll_i = tf.constant([[0,1]],dtype=tf.complex64)
        # Calculation
        ## Gdel
        self.Gdel = self.I*tf.exp(-self.s_3*self.Tdel) # Gdel is the block of the delay caused by control
        ## Hi
        self.Hi = (self.kpi + self.kii/self.s_3)*self.I # Hi is the block of the PI compensator of the current controller
        ## Jw1L
        self.Jw1L = tf.broadcast_to((self.w1*self.Lfilter)*self.J,[self.s_3.shape[0],2,2]) # Jw1L is the block of the decoupling part of the current controller 
        ## Hv
        self.Hv = (self.kpu + self.kiu/self.s_3)*V_i # Hv is the block of the voltage controller     
        ## TpllM
        self.Gpll = (self.kppll + self.kipll/self.s_3)/self.s_3
        self.Tpll = self.Gpll/(1+self.Vsd*self.Gpll)
        self.TpllM = self.Tpll*Tpll_i # TpllM is the block of the small signal model of the PLL
        ## Hpll
        self.Hpll = self.J@self.Dcdq@self.TpllM # Hpll is a block related to PLL
        ## Ypll
        self.Ypll =  self.J@self.Isdq@self.TpllM # Ypll is a block related to PLL
        ## C
        # self.C = 3/2/(self.Cout*self.s_3+1/self.Rl) # When considering the DCDC converter impedance
        self.C = 3/2/(self.Cout*self.s_3) # When neglecting the DCDC converter impedance
        ## Gd2v
        self.Gd2v = self.Vdc*self.I+self.C*self.Dsdq@self.IsdqT # when the DC voltage ripple is considered
        # when neglecting DC voltage ripple, it is Vdc
        Vdcm = self.Vdc*self.I
        self.Vdcm3 = tf.broadcast_to(Vdcm,[self.s_3.shape[0],2,2]) # Vdc with correct dimension
        ## Gd2dc
        self.Gd2dc = self.C*self.IsdqT # when considering DC voltage ripple, the gain from duty cycle to Vdc
        ## Gi2dc
        self.Gi2dc = self.C*self.DsdqT # when considering DC voltage ripple, the gain from current to Vdc

    def cal_impedance(self):     
        # Calculate the impedance of the passive components (i.e. power filter and DC link caps)
        ## Z(s+jw1) is the block of the impedance of the power filter
        self.Z = self.Jw1L + (self.Rfilter + self.Lfilter*self.s_3)*self.I 
        ## Y(s+jw1) is the block of the admitance of the power filter
        self.Y = inv(self.Z) 
        ## impedance w/ DC voltage ripple
        self.Zvd_pas_dq = self.Z + self.C*self.Dsdq@self.DsdqT 
        ## admitance w/ DC voltage ripple
        self.Yvd_pas_dq = inv(self.Zvd_pas_dq) 
        ### There are some simplified expression for Zvd_pas_dq and Yvd_pas_dq
        #### one of the simplified expression of Zvd_pas_dq and Yvd_pas_dq, details in Maple
        M00 = self.Dsdq[0]**2*self.C
        M01 = self.Dsdq[0]*self.Dsdq[1]*self.C
        M0 = tf.concat([M00,M01],axis = 2)
        M10 = M01
        M11 = 0/self.s_3
        M1 = tf.concat([M10,M11],axis = 2)
        M = tf.concat([M0,M1],axis = 1)
        self.Zvd_pas_dq_sim = self.Z + M # a simplified expression of Zvd_pas_dq, details in Maple
        self.Yvd_pas_dq_sim = inv(self.Zvd_pas_dq_sim) # a simplified expression of Yvd_pas_dq, details in Maple
        # define some expression for simplicity
        a = self.Lfilter*self.s_3+self.Rfilter
        b = 1+self.Dsdq[0]*self.Isdq[0]*self.C/self.Vdc
        c = self.Lfilter*self.w1
        d = self.Dsdq[0]*self.Dsdq[1]*self.C
        e = self.Dsdq[1]*self.Isdq[0]*self.C/self.Vdc
        f = self.C*self.Dsdq[0]**2
        den1 = a**2+f*a+c**2
        den2 = a+f
        edel = tf.expand_dims(tf.expand_dims(self.Gdel[:,0,0],axis = 1),axis = 1)
        I_3  = tf.broadcast_to(self.I,[self.s_3.shape[0],2,2])
        ele_I = tf.expand_dims(tf.expand_dims(I_3[:,0,0],axis = 1),axis = 1)
        ele_Hv = tf.expand_dims(tf.expand_dims(self.Hv[:,0,0],axis = 1),axis = 1)
        self.wci = self.kpi/self.Lfilter # cutoff frequency of current loop
        self.Gi_ol_siso = edel*self.wci/self.s_3
        g = self.s_3*self.Vdc/self.C
        h = a*self.wci*self.Isdq[0]*ele_Hv*edel
        den3 = h-g
        #### a further simplified expression of Yvd_pas_dq, details in Maple
        M00 = (self.Lfilter*self.s_3+self.Rfilter)/den1
        M01 = (self.Lfilter*self.w1 - self.Dsdq[0]*self.Dsdq[1]*self.C)/den1
        M10 = -(self.Lfilter*self.w1 + self.Dsdq[0]*self.Dsdq[1]*self.C)/den1
        M11 = (self.Lfilter*self.s_3+self.Rfilter+self.Dsdq[0]**2*self.C)/den1
        M0 = tf.concat([M00,M01],axis = 2)
        M1 = tf.concat([M10,M11],axis = 2)
        self.Yvd_pas_dq_sim2  = tf.concat([M0,M1],axis = 1) # a further simplified expression of Yvd_pas_dq, details in Maple

        # Calculate the impedance of the plant H (i.e. passive components+control delay+gain of modulation)
        ## plant wo/ DC voltage ripple and control delay
        self.Hwo_vd_cd = self.Y
        ## plant wo/ DC voltage ripple but w/ control delay
        self.Hwo_vd = self.Y@self.Gdel 
        ## plant w DC voltage ripple but wo/ control delay
        self.Hwo_cd = 1/self.Vdc*self.Yvd_pas_dq@self.Gd2v
        ### There is also simplified expression for Hwo_cd
        M00 = a*b/den1
        M01 = (c-d)/den1
        M10 = (a*e-c)/den1
        M11 = (a+f)/den1
        M0 = tf.concat([M00,M01],axis = 2)
        M1 = tf.concat([M10,M11],axis = 2)
        self.Hwo_cd_sim  = tf.concat([M0,M1],axis = 1)# simplifier expression of Hwo_cd
        ## plant w/ DC voltage ripple and control delay
        self.H = 1/self.Vdc*self.Yvd_pas_dq@self.Gd2v@self.Gdel 
        ### The expression of H cannot be simplified, which is proven below
        M00 = b*edel/den2
        M01 = (c-d)*edel/a/den2
        M10 = (a*e-c)*edel/a/den2
        M11 = edel/a
        M0 = tf.concat([M00,M01],axis = 2)
        M1 = tf.concat([M10,M11],axis = 2)
        self.Hsim = tf.concat([M0,M1],axis = 1) # With this result, it is shown that H cannot be simplified

        # Calculate the impedance of the plant after decoupling Hde (i.e. H with decoupling loop)
        ## decoupled plant wo/ DC voltage ripple and control delay
        self.Hde_wo_vd_cd = inv(self.I-self.Hwo_vd_cd@self.Jw1L)@self.Hwo_vd_cd 
        ## decoupled plant wo/ DC voltage ripple but w/ control delay
        self.Hde_wo_vd = inv(self.I-self.Hwo_vd@self.Jw1L)@self.Hwo_vd 
        ### The expression of Hde_wo_vd can be simplified
        M00 = 1/(self.Lfilter*self.s_3+self.Rfilter)
        M11 = 1/(self.Lfilter*self.s_3+self.Rfilter)
        M01 = -self.w1*self.Lfilter*(edel-1)\
                /(self.Lfilter*self.s_3+self.Rfilter)**2
        M10 = self.w1*self.Lfilter*(edel-1)\
                /(self.Lfilter*self.s_3+self.Rfilter)**2
        M0 = tf.concat([M00,M01],axis = 2)
        M1 = tf.concat([M10,M11],axis = 2)
        self.Hde_wo_vd_sim = tf.concat([M0,M1],axis = 1)@self.Gdel # this is the simplified version of H_de_wo_vd
        ## decoupled plant w/ DC voltage ripple but wo/ control delay
        self.Hde_wo_cd = inv(self.I-self.Hwo_cd@self.Jw1L)@self.Hwo_cd 
        ### The expression of Hde_wo_cd can be simplified
        M00 = (b*a**2+f*a-c*d+f*(b-1)*a)/den1/den2
        M01 = 0*self.s_3
        M10 = -(-e*a**3+d*a**2+d*c**2-e*f*a**2+d*f*a)/den1/den2/a
        M11 = 1/a
        M0 = tf.concat([M00,M01],axis = 2)
        M1 = tf.concat([M10,M11],axis = 2)
        self.Hde_wo_cd_sim  = tf.concat([M0,M1],axis = 1) # This is the first version of the simplification
        ### The expression can be furthur simplified
        M00 = b/den2
        M01 = 0*self.s_3
        M10 = -(-e*a+d)/a/den2
        M11 = 1/a
        M0 = tf.concat([M00,M01],axis = 2)
        M1 = tf.concat([M10,M11],axis = 2)
        self.Hde_wo_cd_sim2  = tf.concat([M0,M1],axis = 1) # This is the second version of the simplification
        ## decoupled plant w/ DC voltage ripple and control delay
        self.Hde = inv(self.I-self.H@self.Jw1L)@self.H 
        ### The expression of Hde can be simplified
        M00 = b*edel/den2
        M01 = -(edel-1)*(c-d)*edel/(a*den2)
        M10 = (a*e-d+(c-d)*(edel-1))*edel/a/den2
        M11 = edel/a
        M0 = tf.concat([M00,M01],axis = 2)
        M1 = tf.concat([M10,M11],axis = 2)
        self.Hde_sim = tf.concat([M0,M1],axis = 1)

        # Calculate the open loop transfer function of the current loop Gol
        ## TF of the current loop w/ DC voltage ripples and control delay but wo/ decoupled plant
        self.Gol_wode = self.H@self.Hi
        ## TF of the current loop w/ DC voltage ripples but wo/ control delay and decoupled plant
        self.Gol_wo_cd = self.Hde_wo_cd@self.Hi
        ## TF of the current loop wo/ DC voltage ripples but w/ control delay and decoupled plant
        self.Gol_wo_vd = self.Hde_wo_vd@self.Hi
        ### The expression of Gol_wo_vd can be simplified
        self.Gol_wo_vd_sim = self.Hde_wo_vd_sim@self.Hi
        ## TF of the current loop wo/ DC voltage ripples and control delay but w/ decoupled plant
        self.Gol_wo_vd_cd = self.Hde_wo_vd_cd@self.Hi
        ###  For simplicity, we also define Gol_wo_vd_cd_rsd below
        M00 = ele_I
        M01 = -self.w1*self.Lfilter*(edel-1)/(self.Lfilter*self.s_3+self.Rfilter)
        M10 = -M01
        M11 = ele_I
        M0 = tf.concat([M00,M01],axis = 2)
        M1 = tf.concat([M10,M11],axis = 2)
        self.Gol_wo_vd_rsd = tf.concat([M0,M1],axis = 1)
        #### The inverse of Gol_wo_vd_rsd can be simplified, which will be used for the simplification of Zcl_dq_wo_vd
        M00 = ele_I
        M01 = self.w1*self.Lfilter*(edel-1)/(self.Lfilter*self.s_3+self.Rfilter)
        M10 = -M01
        M11 = ele_I
        M0 = tf.concat([M00,M01],axis = 2)
        M1 = tf.concat([M10,M11],axis = 2)
        self.Gol_wo_vd_rsd_inv_sim = tf.concat([M0,M1],axis = 1)
        ## TF of the current loop w/ DC voltage ripples, control delay and decoupled plant
        self.Gol = self.Hde@self.Hi
        ### For simplicity, we also define Gol_rsd below
        M00 = b*a/den2
        M01 = -(c-d)*(edel-1)/den2
        M10 = (a*e-c+(c-d)*edel)/den2
        M11 = ele_I
        M0 = tf.concat([M00,M01],axis = 2)
        M1 = tf.concat([M10,M11],axis = 2)
        self.Gol_rsd = tf.concat([M0,M1],axis = 1)
        #### The inverse of Gol_sim_rsd can be simplified, which will be used for the simplification of Zcl_dq
        M00 = den2**2/(b*a*den2)
        M01 = (c-d)*(edel-1)*den2/(b*a*den2)
        M10 = -(a*e-c+(c-d)*edel)*den2/(b*a*den2)
        # M11 = 1- ((c-d)**2*(edel-1)**2+(a*e-d)*(c-d)*(edel-1))/(b*a*den2) # This is the expression without error
        M11 = ele_I # Not sure if this simplification will result in error or not
        M0 = tf.concat([M00,M01],axis = 2)
        M1 = tf.concat([M10,M11],axis = 2)
        self.Gol_rsd_inv_sim = tf.concat([M0,M1],axis = 1)

        # Calculate the gain Ge2i
        ## Ge2i w/ DC voltage ripples but wo/ voltage control loop
        Ge2i_1 = self.Vdc*inv(self.Hi)@inv(self.Gd2v@self.Gdel)
        Ge2i_2 = -self.Vdc*inv(self.Hi)@self.Hpll
        Ge2i_3 = self.Ypll
        Ge2i_4 = -inv(self.Hi)@self.Jw1L@self.Ypll
        self.Ge2i = Ge2i_1+Ge2i_2+Ge2i_3+Ge2i_4
        ### For simplicity, we also define Ge2i_rsd below
        self.Ge2i_rsd = self.Gi_ol_siso*self.Ge2i
        ## Ge2i wo/ DC voltage ripples and voltage control loop
        Ge2i_wo_vd_1 = inv(self.Hi)@inv(self.Gdel)
        Ge2i_wo_vd_2 = -self.Vdc*inv(self.Hi)@self.Hpll
        Ge2i_wo_vd_3 = self.Ypll
        Ge2i_wo_vd_4 = -inv(self.Hi)@self.Jw1L@self.Ypll
        self.Ge2i_wo_vd = Ge2i_wo_vd_1+Ge2i_wo_vd_2+Ge2i_wo_vd_3+Ge2i_wo_vd_4
        ## Ge2i wo/ DC voltage ripples, voltage control loop and control delay
        self.Ge2i_wo_vd_cd = I_3+(self.Hi - self.Jw1L)@self.Ypll-self.Vdc*self.Hpll

        # Calculate the block Gi2dc_tot
        self.Gi2dc_tot = self.Gi2dc-self.Gd2dc@inv(self.Gd2v)@self.Zvd_pas_dq
        ## Simplification of Gi2dc_tot
        M00 = (self.Dsdq[0]*self.Vdc-self.Isdq[0]*a)*self.C/self.Vdc/b
        M01 = 0*ele_I
        self.Gi2dc_tot_sim = tf.concat([M00,M01],axis = 2)

        # Calculate the block Ge2i_tot
        self.Ge2i_tot = self.Ge2i-self.Hv@self.Gd2dc@inv(self.Gd2v)
        ## we define Ge2i_tot_2 to check if Ge2i_tot can be simplified or not
        Spll = tf.expand_dims(tf.expand_dims(self.Ge2i[:,1,1],axis = 1),axis = 1)
        M00 = self.C*ele_Hv*self.Isdq[0]/self.Vdc/b
        M01 = 0*ele_I
        M10 = 0*ele_I
        M11 = M10
        M0 = tf.concat([M00,M01],axis = 2)
        M1 = tf.concat([M10,M11],axis = 2)
        self.Ge2i_tot_2_sim = tf.concat([M0,M1],axis = 1)
        ## The inverse of Ge2i_tot can be simplified as 
        M00 = -edel*b*a*self.wci*g/self.s_3/den3
        M01 = 0*ele_I
        M10 = -e*g/Spll/den3
        M11 = 1/Spll
        M0 = tf.concat([M00,M01],axis = 2)
        M1 = tf.concat([M10,M11],axis = 2)
        self.Ge2i_tot_inv_sim  = tf.concat([M0,M1],axis = 1)

        # Calculate the impedance wo/ current loop, voltage loop, and voltage ripples
        self.Zpll_dq_wo_vd = inv(self.I-self.Vdc*self.Gdel@self.Hpll)@self.Z

        # Calculate the impedance wo/ current and voltage loop but with voltage ripples
        self.Zpll_dq = inv(self.I-self.Gd2v@self.Gdel@self.Hpll)@self.Zvd_pas_dq 

        # Calculate the impedance wo/ voltage control loop, DC voltage ripples and control delay
        self.Zcl_dq_wo_vd_cd = inv(self.Ge2i_wo_vd_cd)@(self.I+inv(self.Gol_wo_vd_cd)) # dq-frame
        self.Zcl_sdq_wo_vd_cd,self.Zcl_ab_wo_vd_cd = self.convert_dq2ab(self.Zcl_dq_wo_vd_cd)# ab-frame 

        # Calculate the impedance wo/ voltage control loop but w/ DC voltage ripples
        self.Zcl_dq = inv(self.Ge2i)@(self.I+inv(self.Gol)) # dq-frame
        self.Zcl_dq_sim = inv(self.Ge2i_rsd)@(self.Gi_ol_siso*self.I+self.Gol_rsd_inv_sim)
        self.Zcl_sdq,self.Zcl_ab = self.convert_dq2ab(self.Zcl_dq)# ab-frame

        # Calculate the impedance wo/ voltage control loop and DC voltage ripples
        self.Zcl_dq_wo_vd = inv(self.Ge2i_wo_vd)@(self.I+inv(self.Gol_wo_vd))
        self.Zcl_dq_wo_vd_sim = 1/self.Gi_ol_siso*inv(self.Ge2i_wo_vd)@(self.Gol_wo_vd_rsd_inv_sim+self.Gi_ol_siso*I_3)
        self.Zcl_sdq_wo_vd,self.Zcl_ab_wo_vd = self.convert_dq2ab(self.Zcl_dq_wo_vd)

        # Calculate the impedance when considering the dynamics of the PLL, CL, and VL
        self.Zvl_dq = inv(self.Ge2i_tot)@(self.I+inv(self.Gol)+self.Hv@self.Gi2dc_tot)
        self.Zvl_dq_sim = self.Ge2i_tot_inv_sim@(self.I+1/self.Gi_ol_siso*self.Gol_rsd_inv_sim+self.Hv@self.Gi2dc_tot_sim)
        self.Zvl_sdq,self.Zvl_ab = self.convert_dq2ab(self.Zvl_dq)

        # Another approach: Calculate the impedance when considering the dynamics of the PLL, CL, and VL
        self.K11 = self.Vdc*self.I+self.C*self.Dsdq@self.IsdqT
        self.K12 = inv(-self.Vdc*inv(self.Gdel)+self.C*self.Hi@self.Hv@self.IsdqT)
        self.K13 = (self.Hi-self.Jw1L)@self.Ypll-self.Vdc*self.Hpll
        self.K1 = self.I - self.K11@self.K12@self.K13
        self.K2 = self.Zvd_pas_dq
        self.K31 = self.K11
        self.K32 = self.K12
        self.K33 = self.Hi-self.Jw1L+self.C*self.Hi@self.Hv@self.DsdqT
        self.K3 =  self.K31@self.K32@self.K33
        self.Zvl_dq_v2 = inv(self.K1)@(self.K2-self.K3)
        self.Zvl_sdq_v2,self.Zvl_ab_v2 = self.convert_dq2ab(self.Zvl_dq_v2)

    def convert_dq2ab(self,Z_dq):
        # Calculate the impedance in the ab frame
        Az1 = np.array([[1/2,1j/2],[1/2,-1j/2]])
        Az2 = np.array([[1,1],[-1j,1j]])
        Az1_ts = tf.constant([Az1],dtype=tf.complex64)
        Az2_ts = tf.constant([Az2],dtype=tf.complex64)
        Zsdq = Az1_ts@Z_dq@Az2_ts #[[Z+dq,Z-dq],
                                # [Z-dq_conj,Z+dq_conj]]
        Zsdq_n,Zsdq_p = tf.split(Zsdq,2,axis=0)

        Zsdq_p_det = tf.linalg.det(Zsdq_p)
        Zsdq_n_det = tf.linalg.det(Zsdq_n)

        Zp_dq_p_conj = Zsdq_p[:,1,1]
        Zn_dq_p_conj = Zsdq_p[:,1,0]
        Zp = tf.expand_dims(Zsdq_p_det/Zp_dq_p_conj,axis=1)
        Zpc = tf.expand_dims(tf.math.conj(-Zsdq_p_det/Zn_dq_p_conj),axis=1)

        Zp_dq_n_conj = Zsdq_n[:,1,1]
        Zn_dq_n_conj = Zsdq_n[:,1,0]
        Zn = tf.expand_dims(Zsdq_n_det/Zp_dq_n_conj,axis=1)
        Znc = tf.expand_dims(tf.math.conj(-Zsdq_n_det/Zn_dq_n_conj),axis=1)
        Zppc = tf.expand_dims(tf.concat([Zp,Zpc],1),axis=1)
        Zncn = tf.expand_dims(tf.concat([Znc,Zn],1),axis=1)
        Z_ab = tf.concat([Zppc,Zncn],1)
        return Zsdq,Z_ab
    
    


#***********************************************#
#----------------End Model 1 -------------------#
#***********************************************#


