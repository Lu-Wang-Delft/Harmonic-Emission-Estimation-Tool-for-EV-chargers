import tensorflow as tf
import numpy as np
from tensorflow.linalg import inv
#***********************************************#
#------------------ Model 1 --------------------#
#***********************************************#
class Impedance_model1():  # The design of this model is: CC(SRF-PI)+PLL(SRF-PLL)+DVC(PI)+filter(L-)+modulation(SPWM)+topology(2L or Vienna)
    def __init__(self, Rfilter, Lfilter,kpi,kii, kppll, kipll, kpu,kiu,fsw,
                Cout,Vdc,f1,Vsdq,Power,f_sweep): # The input dtype is float32
                      
        self.Rfilter = tf.cast(Rfilter,dtype=tf.complex64)
        self.Lfilter = tf.cast(Lfilter,dtype=tf.complex64)
        self.kpi = tf.cast(kpi,dtype=tf.complex64)
        self.kii = tf.cast(kii,dtype=tf.complex64)
        self.kppll = tf.cast(kppll,dtype=tf.complex64)
        self.kipll = tf.cast(kipll,dtype=tf.complex64)
        self.kpu = tf.cast(kpu,dtype=tf.complex64)
        self.kiu = tf.cast(kiu,dtype=tf.complex64)
        self.fsw = tf.cast(fsw,dtype=tf.complex64)
        self.Cout = tf.cast(Cout/2,dtype=tf.complex64)
        self.Vdc = tf.cast(Vdc,dtype=tf.complex64)
        self.Vsd = Vsdq[0]
        self.Vsq = Vsdq[1]
        self.Pi = tf.constant(np.pi,dtype=tf.complex64)
        self.w1 = 2*self.Pi*f1
        self.P = Power

        # Calculate the delay of the control
        self.Tdel = 1.5/self.fsw
        # Calculate the s domain
        self.S_domain(f_sweep)
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
        ## plant wo/ DC voltage ripple but w/ control delay
        self.Hwo_vd = self.Y@self.Gdel 
        ## decoupled plant wo/ DC voltage ripple but w/ control delay
        self.Hde_wo_vd = inv(self.I-self.Hwo_vd@self.Jw1L)@self.Hwo_vd 
        ## TF of the current loop wo/ DC voltage ripples but w/ control delay and decoupled plant
        self.Gol_wo_vd = self.Hde_wo_vd@self.Hi
        ## Ge2i wo/ DC voltage ripples and voltage control loop
        Ge2i_wo_vd_1 = inv(self.Hi)@inv(self.Gdel)
        Ge2i_wo_vd_2 = -self.Vdc*inv(self.Hi)@self.Hpll
        Ge2i_wo_vd_3 = self.Ypll
        Ge2i_wo_vd_4 = -inv(self.Hi)@self.Jw1L@self.Ypll
        self.Ge2i_wo_vd = Ge2i_wo_vd_1+Ge2i_wo_vd_2+Ge2i_wo_vd_3+Ge2i_wo_vd_4
        # Calculate the impedance wo/ voltage control loop and DC voltage ripples
        self.Zcl_dq_wo_vd = inv(self.Ge2i_wo_vd)@(self.I+inv(self.Gol_wo_vd))
        self.Zcl_ab_wo_vd = self.convert_dq2ab(self.Zcl_dq_wo_vd)

        # Calculate the impedance when considering the dynamics of the PLL, CL, and VL
        self.K11 = self.Vdc*self.I+self.C*self.Dsdq@self.IsdqT
        self.K12 = inv(-self.Vdc*inv(self.Gdel)+self.C*self.Hi@self.Hv@self.IsdqT)
        self.K13 = (self.Hi-self.Jw1L)@self.Ypll-self.Vdc*self.Hpll
        self.K1 = self.I - self.K11@self.K12@self.K13
        self.K2 = self.Zvd_pas_dq
        self.K31 = self.K11
        self.K32 = self.K12
        self.K33 = self.Hi-self.Jw1L+self.C*self.Hi@self.Hv@self.DsdqT
        self.K3 =  self.K31@self.K32@self.K33
        self.Zvd_vl_dq = inv(self.K1)@(self.K2-self.K3)
        self.Zvd_vl_ab = self.convert_dq2ab(self.Zvd_vl_dq)
    def convert_dq2ab(self,Z_dq):
        # Calculate the impedance in the ab frame
        Az1 = np.array([[1/2,-1j/2],[1/2,1j/2]])
        Az2 = np.array([[1,1],[1j,-1j]])
        Az1_ts = tf.constant([Az1],dtype=tf.complex64)
        Az2_ts = tf.constant([Az2],dtype=tf.complex64)
        Zsdq = Az1_ts@Z_dq@Az2_ts #[[Z+dq_conj,Z-dq_conj],
                                    # [Z-dq     ,Z+dq]]
        Zsdq_n,Zsdq_p = tf.split(Zsdq,2,axis=0)

        Zsdq_p_det = tf.linalg.det(Zsdq_p)
        Zsdq_n_det = tf.linalg.det(Zsdq_n)

        Zp_dq_p_conj = Zsdq_p[:,0,0]
        Zn_dq_p_conj = Zsdq_p[:,0,1]
        Zp = tf.expand_dims(Zsdq_p_det/Zp_dq_p_conj,axis=1)
        Zpc = tf.expand_dims(tf.math.conj(-Zsdq_p_det/Zn_dq_p_conj),axis=1)

        Zp_dq_n_conj = Zsdq_n[:,0,0]
        Zn_dq_n_conj = Zsdq_n[:,0,1]
        Zn = tf.expand_dims(Zsdq_n_det/Zp_dq_n_conj,axis=1)
        Znc = tf.expand_dims(tf.math.conj(-Zsdq_n_det/Zn_dq_n_conj),axis=1)
        Zppc = tf.expand_dims(tf.concat([Zp,Zpc],1),axis=1)
        Zncn = tf.expand_dims(tf.concat([Znc,Zn],1),axis=1)
        Z_ab = tf.concat([Zppc,Zncn],1)
        return Z_ab
    
#***********************************************#
#----------------End Model 1 -------------------#
#***********************************************#


