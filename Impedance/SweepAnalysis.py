import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv

def Impedance_pnsweep(fsweep_sim,folder):
    Zsweep = np.zeros([fsweep_sim.shape[0],2,2],dtype=complex)
    Ysweep = np.zeros([fsweep_sim.shape[0],2,2],dtype=complex)
    for idx,fsweep in enumerate(fsweep_sim):
        # Positive sequence injection results
        fileend = str(fsweep)
        filename = folder+'\\Upp'+fileend+'.csv'
        name = ['Time','Upd','Upq','Ipd','Ipq','Icd','Icq']
        data = pd.read_csv(filename,encoding= 'unicode_escape',names=name,header=0)
        Up,Ip,Ipc = FT_calpn(data)
        # Negative sequence injection results
        filename = folder+'\\Upn'+fileend+'.csv'
        data = pd.read_csv(filename,encoding= 'unicode_escape',names=name,header=0)
        Un,In,Inc = FT_calpn(data)

        Zsweep[idx,0,0] = Up/Ip
        Zsweep[idx,0,1] = Up/Ipc
        Zsweep[idx,1,0] = Un/Inc
        Zsweep[idx,1,1] = Un/In
        Ysweep[idx,:,:] = inv(Zsweep[idx,:,:])
    return Zsweep, Ysweep

def FT_calpn(data):
    t = data['Time'].to_numpy()
    Updt = data['Upd'].to_numpy()
    Upqt = data['Upq'].to_numpy()
    Ipdt = data['Ipd'].to_numpy()
    Ipqt = data['Ipq'].to_numpy()
    Icdt = data['Icd'].to_numpy()
    Icqt = data['Icq'].to_numpy()
    L = data.shape[0]
    T = t[-1] - t[0]
    F = np.zeros([6,])
    # Only calculate the DC components
    for m in range(L-1):
        F[0] = F[0] + Updt[m]*(t[m+1] - t[m])
        F[1] = F[1] + Upqt[m]*(t[m+1] - t[m])
        F[2] = F[2] + Ipdt[m]*(t[m+1] - t[m])
        F[3] = F[3] + Ipqt[m]*(t[m+1] - t[m])
        F[4] = F[4] + Icdt[m]*(t[m+1] - t[m])
        F[5] = F[5] + Icqt[m]*(t[m+1] - t[m])

    Upd = F[0]/T
    Upq = F[1]/T
    Ipd = F[2]/T
    Ipq = F[3]/T
    Icd = F[4]/T
    Icq = F[5]/T

    Up = Upd+1j*Upq
    Ip = Ipd+1j*Ipq
    Ic = Icd+1j*Icq
    return Up,Ip,Ic

def ft_vs(t,S,fbase,n):
    w0 = fbase*2*np.pi
    L = t.shape[0]
    T = t[-1] - t[0]
    F = np.zeros([n+1,],dtype=complex)
    F_mag = np.zeros([n+1,])
    F_phi = np.zeros([n+1,])
    for m in range(L-1):
        F[0] = F[0] + S[m]*(t[m+1] - t[m])
    F[0] = F[0]/T
    F_mag[0] = np.abs(F[0])
    F_phi[0] = 0
    for m in range(L-1):
        for i in range(1,n+1):
            F[i] = F[i] + S[m]/-1j/w0/i * (np.exp(-1j*w0*i*t[m+1]) - np.exp(-1j*w0*i*t[m]))
    F = 2/T*F
    F_mag = np.abs(F)
    F_phi = np.angle(F,deg=True)
    f_sweep = fbase*np.arange(n+1)
    return f_sweep,F_mag,F_phi

def Impedance_dqsweep(fsweep_sim,folder):
    Zsweep = np.zeros([fsweep_sim.shape[0],2,2],dtype=complex)
    Ysweep = np.zeros([fsweep_sim.shape[0],2,2],dtype=complex)
    for idx,fsweep in enumerate(fsweep_sim):
        # d-axis injection results
        fileend = str(fsweep)
        filename = folder+'\\Upd'+fileend+'.csv'
        name = ['Time','Ud','Uq','Id','Iq']
        data = pd.read_csv(filename,encoding= 'unicode_escape',names=name,header=0)
        Ud,Uq,Id,Iq = FT_caldq(data,fsweep)
        # q-axis injection results
        filename = folder+'\\Upq'+fileend+'.csv'
        data = pd.read_csv(filename,encoding= 'unicode_escape',names=name,header=0)
        Ud_,Uq_,Id_,Iq_ = FT_caldq(data,fsweep)

        Zsweep[idx,:,:] = np.array([[Ud,Ud_],[Uq,Uq_]])@inv(np.array([[Id,Id_],[Iq,Iq_]]))
        Ysweep[idx,:,:] = inv(Zsweep[idx,:,:])
    return Zsweep, Ysweep

def Impedance_dqsweep_switching(fsweep_sim,folder):
    Zsweep = np.zeros([fsweep_sim.shape[0],2,2],dtype=complex)
    Ysweep = np.zeros([fsweep_sim.shape[0],2,2],dtype=complex)
    for idx,fsweep in enumerate(fsweep_sim):
        fileend = str(fsweep)
        name = ['Time','Ud','Uq','Id','Iq']
        # reference
        filename = folder+'\\Uref'+'.csv'
        data = pd.read_csv(filename,encoding= 'unicode_escape',names=name,header=0)
        Ud_ref,Uq_ref,Id_ref,Iq_ref = FT_caldq(data,fsweep)
        # d-axis injection results
        filename = folder+'\\Upd'+fileend+'.csv'
        data = pd.read_csv(filename,encoding= 'unicode_escape',names=name,header=0)
        Ud,Uq,Id,Iq = FT_caldq(data,fsweep)
        # q-axis injection results
        filename = folder+'\\Upq'+fileend+'.csv'
        data = pd.read_csv(filename,encoding= 'unicode_escape',names=name,header=0)
        Ud_,Uq_,Id_,Iq_ = FT_caldq(data,fsweep)
        # calculate the incremental harmonics
        Ud = Ud - Ud_ref
        Uq = Uq - Uq_ref
        Id = Id - Id_ref
        Iq = Iq - Iq_ref
        Ud_ = Ud_ - Ud_ref
        Uq_ = Uq_ - Uq_ref
        Id_ = Id_ - Id_ref
        Iq_ = Iq_ - Iq_ref

        Zsweep[idx,:,:] = np.array([[Ud,Ud_],[Uq,Uq_]])@inv(np.array([[Id,Id_],[Iq,Iq_]]))
        Ysweep[idx,:,:] = inv(Zsweep[idx,:,:])
    return Zsweep, Ysweep

def FT_caldq(data,f):
    t = data['Time'].to_numpy()
    Udt = data['Ud'].to_numpy()
    Uqt = data['Uq'].to_numpy()
    Idt = data['Id'].to_numpy()
    Iqt = data['Iq'].to_numpy()
    w = f*2*np.pi

    L = data.shape[0]
    T = t[-1] - t[0]
    F = np.zeros([4,],dtype=complex)
    # Only calculate the DC components
    for m in range(L-1):
        F[0] = F[0] + Udt[m]/-1j/w * (np.exp(-1j*w*t[m+1]) - np.exp(-1j*w*t[m]))
        F[1] = F[1] + Uqt[m]/-1j/w * (np.exp(-1j*w*t[m+1]) - np.exp(-1j*w*t[m]))
        F[2] = F[2] + Idt[m]/-1j/w * (np.exp(-1j*w*t[m+1]) - np.exp(-1j*w*t[m]))
        F[3] = F[3] + Iqt[m]/-1j/w * (np.exp(-1j*w*t[m+1]) - np.exp(-1j*w*t[m]))

    Ud = 2/T*F[0]
    Uq = 2/T*F[1]
    Id = 2/T*F[2]
    Iq = 2/T*F[3]

    return Ud,Uq,Id,Iq

def FreqResponse(fsweep_sim,folder):
    G_id = np.zeros([fsweep_sim.shape[0],],dtype=complex)
    G_iq = np.zeros([fsweep_sim.shape[0],],dtype=complex)
    G_u = np.zeros([fsweep_sim.shape[0],],dtype=complex)
    name = ['Time','Before','After']
    for idx,fsweep in enumerate(fsweep_sim):
        filename = folder+'.csv'
        fileend = str(fsweep)
        # Id open loop transfer function    
        filename = folder+'\\Ipd'+fileend+'.csv'
        data = pd.read_csv(filename,encoding= 'unicode_escape',names=name,header=0)
        F_id = ft_f(data,fsweep)
        G_id[idx] = F_id[0]/-F_id[1]
        # Iq open loop transfer function    
        filename = folder+'\\Ipq'+fileend+'.csv'
        data = pd.read_csv(filename,encoding= 'unicode_escape',names=name,header=0)
        F_iq = ft_f(data,fsweep)
        G_iq[idx] = F_iq[0]/-F_iq[1]
        # Vdc open loop transfer function    
        filename = folder+'\\Up'+fileend+'.csv'
        data = pd.read_csv(filename,encoding= 'unicode_escape',names=name,header=0)
        F_u = ft_f(data,fsweep)
        G_u[idx] = F_u[0]/-F_u[1]
    return G_id,G_iq,G_u

def ft_f(data,f):
    t = data['Time'].to_numpy()
    BI = data['Before'].to_numpy()
    AI = data['After'].to_numpy()
    w0 = f*2*np.pi
    L = t.shape[0]
    T = t[-1] - t[0]
    F = np.zeros([2,],dtype=complex)
    if f==0:
        for m in range(L-1):
            F[0] = F[0] + BI[m]*(t[m+1] - t[m])
            F[1] = F[1] + AI[m]*(t[m+1] - t[m])
        F = F/T
    else:
        for m in range(L-1):
            F[0] = F[0] + BI[m]/-1j/w0 * (np.exp(-1j*w0*t[m+1]) - np.exp(-1j*w0*t[m]))
            F[1] = F[1] + AI[m]/-1j/w0 * (np.exp(-1j*w0*t[m+1]) - np.exp(-1j*w0*t[m]))
        F = 2/T*F
    return F

def Impedance_plotting(fsweep,fsim,num_sim,title,x_lim=[0,1500],*mark,**Ana_Im):
    fig, subaxes = plt.subplots(4, 2, figsize=(11,11), dpi=80)
    for imk,(key,Ana_Z) in enumerate(Ana_Im.items()):
        # if imk != 0:
        #     Ana_Z[fsweep==0] = np.nan
        Ana_Z_dd = Ana_Z[:,0,0]
        Ana_Z_dq = Ana_Z[:,0,1]
        Ana_Z_qd = Ana_Z[:,1,0]
        Ana_Z_qq = Ana_Z[:,1,1]

        Mgdd = 20*np.log10(np.abs(Ana_Z_dd))
        Phdd = np.angle(Ana_Z_dd, deg=True)
        Mgdq = 20*np.log10(np.abs(Ana_Z_dq))
        Phdq = np.angle(Ana_Z_dq, deg=True)
        Mgqd = 20*np.log10(np.abs(Ana_Z_qd))
        Phqd = np.angle(Ana_Z_qd, deg=True)
        Mgqq = 20*np.log10(np.abs(Ana_Z_qq))
        Phqq = np.angle(Ana_Z_qq, deg=True)
        
        Z_data = np.array([Mgdd,Mgdq,Phdd,Phdq,Mgqd,Mgqq,Phqd,Phqq]).T
        if imk < len(Ana_Im.items())-num_sim:
            df = pd.DataFrame(Z_data,columns=['Magnitude of M(1,1)','Magnitude of M(1,2)',
                                                'Phase of M(1,1)','Phase of M(1,2)',
                                                'Magnitude of M(2,1)','Magnitude of M(2,2)',
                                                'Phase of M(2,1)','Phase of M(2,2)'],index=fsweep)
        else:
            df = pd.DataFrame(Z_data,columns=['Magnitude of M(1,1)','Magnitude of M(1,2)',
                                                'Phase of M(1,1)','Phase of M(1,2)',
                                                'Magnitude of M(2,1)','Magnitude of M(2,2)',
                                                'Phase of M(2,1)','Phase of M(2,2)'],index=fsim)
        
        for ix, thisaxisrow in enumerate(subaxes):
            for iy, thisaxis in enumerate(thisaxisrow):
                thisaxis.set_title('{}'.format(df.iloc[:,iy+ix*2].name))
                thisaxis.plot(df.index,df.iloc[:,iy+ix*2],mark[imk],label= key)
                thisaxis.set_xlim(x_lim[0],x_lim[1])
                thisaxis.grid(True)
                thisaxis.set_xlabel('Frequency [Hz]')
                if ix%2==0:
                    thisaxis.set_ylabel('Magnitude [dB]')
                else:
                    thisaxis.set_ylabel('Phase [deg]')
                thisaxis.legend()
    fig.suptitle(title)
    plt.tight_layout()