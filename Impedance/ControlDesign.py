import numpy as np
import matplotlib.pyplot as plt

def design_ctrl(fcPLL,fci,fcv,Lfilter,Rfilter,Vsd,Cout,Vdc,c1,c2):
    # PI controller of PLL
    Kppll = 2*np.pi*fcPLL/Vsd
    Kipll = 2*np.pi*fcPLL*Kppll

    # PI controller of current controller
    wc = fci*2*np.pi
    # wccl = 1.2*wc
    Kpi = Lfilter*wc*c1 #22.5
    Kii = Rfilter*Kpi/Lfilter #1125.16

    # PI controller of the voltage controller
    K = 2*3*Vsd/Cout/Vdc
    # wv = 2*np.pi*fcv; # Bandwidth of voltage controller in rad/s
    # Pu = ((1/fsw*wv)**2+1)*(wv**2+wccl**2)
    # Qu = np.tan(phi_u+np.arctan2(wv,wccl)) # +np.arctan2(wv/fsw,1)
    # Kiu = wv**2/K/wccl*np.sqrt(Pu/(Qu**2+1))# Gain of PI controller 522.25
    # Kpu = Qu/wv*Kii# Integrator of PI controller 0.69
    tau_V = 50*(1+1/fci); # Kpu/Kiu
    wv = 2*np.pi*fcv # Bandwidth of voltage controller in rad/s
    Kiu = wv**2/K*np.sqrt(1/((tau_V*wv)**2+1))*5000*c2 # Gain of PI controller
    Kpu = tau_V*wv/K*np.sqrt(1/((tau_V*wv)**2+1))*5000*c2 # Integrator of PI controller
    # Define the frequency span
    print('Kpi:%f,\nKii:%f,\nKpu:%f,\nKiu:%f.'%(Kpi,Kii,Kpu,Kiu))
    return Kppll,Kipll,Kpi,Kii,Kpu,Kiu

def bode_plot(Rfilter,Lfilter,Kppll,Kipll,Kpi,Kii,Kpu,Kiu,Vsd,Vdc,Cout,fsw,f_sweep):
    f_loop = f_sweep
    s = 2j*np.pi*f_loop
    Gpll = (Kppll + Kipll/s)/s
    PLL_ol = Vsd*Gpll
    PLL_cl = PLL_ol/(1+PLL_ol)
    Y_ideal = 1/(Rfilter+Lfilter*s)
    CL_ol_ideal = (Kpi+Kii/s)*Y_ideal*np.exp(-s*1.5/fsw)
    # CL_ol_ideal_wod = (Kpi+Kii/s)*Y_ideal # without delay
    CL_cl_ideal = CL_ol_ideal/(1+CL_ol_ideal)
    # CL_cl_ideal_wod = CL_ol_ideal_wod/(1+CL_ol_ideal_wod)
    VL_ol_ideal = (Kpu+Kiu/s)*CL_cl_ideal*3*Vsd/Vdc/Cout/s#*np.exp(-s*Ana_model.Tdel)
    VL_cl_ideal = VL_ol_ideal/(1+VL_ol_ideal)

    # Mg_Y_ideal = 20*np.log10(np.abs(Y_ideal))
    # Ph_Y_ideal = np.angle(Y_ideal, deg=True)
    Mg_PLL_ol = 20*np.log10(np.abs(PLL_ol))
    Ph_PLL_ol = np.angle(PLL_ol, deg=True)
    Mg_PLL_cl = 20*np.log10(np.abs(PLL_cl))
    Ph_PLL_cl = np.angle(PLL_cl, deg=True)
    Mg_CL_ol_ideal = 20*np.log10(np.abs(CL_ol_ideal))
    Ph_CL_ol_ideal = np.angle(CL_ol_ideal, deg=True)
    Mg_CL_cl_ideal = 20*np.log10(np.abs(CL_cl_ideal))
    Ph_CL_cl_ideal = np.angle(CL_cl_ideal, deg=True)
    Mg_VL_ol_ideal = 20*np.log10(np.abs(VL_ol_ideal))
    Ph_VL_ol_ideal = np.angle(VL_ol_ideal, deg=True)
    Mg_VL_cl_ideal = 20*np.log10(np.abs(VL_cl_ideal))
    Ph_VL_cl_ideal = np.angle(VL_cl_ideal, deg=True)

    # CL_ol_act2 = 2*np.pi*800/(s+2*np.pi*800)#@(Ana_model.Hi)#(Ana_model.Hi+Ana_model.Jw1L)
    # CL_ol_act = CL_ol_act[:,0,0]
    # CL_ol_act1,CL_ol_act2 = np.split(CL_ol_act,2)
    # Mg_CL_ol_act = 20*np.log10(np.abs(CL_ol_act2))
    # Ph_CL_ol_act = np.angle(CL_ol_act2, deg=True)


    fig,axes= plt.subplots(2,1)
    for ix,ax in enumerate(axes):
        if ix%2 == 0:
            # ax.semilogx(f_loop,Mg_Y_ideal,label = 'Power filter')
            ax.semilogx(f_loop,Mg_PLL_ol,label = 'PLL (OL)')
            ax.semilogx(f_loop,Mg_PLL_cl,label = 'PLL (CL)')
            ax.semilogx(f_loop,Mg_CL_ol_ideal,label = 'Current loop (OL)')
            ax.semilogx(f_loop,Mg_CL_cl_ideal,label = 'Current loop (CL)')
            ax.semilogx(f_loop,Mg_VL_ol_ideal,label = 'Voltage loop (OL)')
            ax.semilogx(f_loop,Mg_VL_cl_ideal,label = 'Voltage loop (CL)')
            ax.set_ylabel('Manitude [dB]')
        else:
            # ax.semilogx(f_loop,Ph_Y_ideal,label = 'Power filter')
            ax.semilogx(f_loop,Ph_PLL_ol,label = 'PLL (OL)')
            ax.semilogx(f_loop,Ph_PLL_cl,label = 'PLL (CL)')
            ax.semilogx(f_loop,Ph_CL_ol_ideal,label = 'Current loop (OL)')
            ax.semilogx(f_loop,Ph_CL_cl_ideal,label = 'Current loop (CL)')
            ax.semilogx(f_loop,Ph_VL_ol_ideal,label = 'Voltage loop (OL)')
            ax.semilogx(f_loop,Ph_VL_cl_ideal,label = 'Voltage loop (CL)')
            ax.set_ylabel('Phase [deg]') 
            ax.set_xlabel('Frequency [Hz]')
        ax.legend()
        ax.grid()
    fig.suptitle('System response')
    plt.show()
    # Show the cut-off frequency and phase margin
    # print('The cut-off frequency of the current loop is %d Hz,\
    #         \nThe phase margin of the current loop is %f deg,\
    #         \nThe cut-off frequency of the voltage loop is %d Hz,\
    #         \nThe phase margin of the voltage loop is %f deg.'
    #         %(f_loop[np.argmin(np.abs(Mg_CL_ol_ideal-1))],
    #         180+Ph_CL_ol_ideal[np.argmin(np.abs(Mg_CL_ol_ideal-1))],
    #         f_loop[np.argmin(np.abs(Mg_VL_ol_ideal-1))],
    #         180+Ph_VL_ol_ideal[np.argmin(np.abs(Mg_VL_ol_ideal-1))]))
