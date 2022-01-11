# Harmonic-Emission-Estimation Tool for Fast-DC-EV-chargers
## Background
Fast charging stations (FCSs) are crucial for the rollout of Electric Vehicles (EVs). However, they, as a power-electronic based non-linear load, have impact on the power quality of the grid. To maintain the compatibility of the power grid with the upcoming massive introduction of FCSs, we need to investigate the potential power quality problems induced by the DC fast chargers. Among all the power quality issues, the harmonic emission noncompliance is one of the most important problems possibly induced by FCSs. 

The harmonic emission estimation tool is thereby developed to provide a cloud based web app that can be used to
- estimate the harmonic emission of a charging station at the design stage when the chargers' design specifications are known,
- estimate the harmonic emission of a charging station at the design stage when the chargers' design specifications are unknown but the chargers' impedance and harmonic current source is measured at certain charging power,
- and estimate the harmonic emission online when the charging station is operating.

The concept of the tool is illustrated below.
<p align="center">
  <img src="./READMEimg/concept.PNG" alt="Scenarios Tree" width="100%" href="#"/>
</p>

The corresponding impedance based model of the concept is illustrated below, where
* I<sub>ci</sub> is the harmonic current source of the charger i,
* Z<sub>ci</sub> is the impedance of the charger i,
* I<sub>ei</sub> is the harmonic emission of the charger i,
* V<sub>PCC</sub> is the voltage at the point-of-common-coupling (PCC), which is the LV AC bus in the figure above,
* I<sub>e,tot</sub> is the harmonic emission of the FCS, 
* Z<sub>g</sub> is the grid impedance
* V<sub>g</sub> is the grid voltage which consists of the fundamental voltage (i.e., V<sub>1</sub>) and the harmonic voltages (i.e., V<sub>h</sub>).

<p align="center">
  <img src="./READMEimg/Impedance_illustration.PNG" alt="Scenarios Tree" width="100%" href="#"/>
</p>

## The considered topology and control strategies
Although the goal is making the tool generally applicable, a typical design is considered at this stage. More topologies and control strategies will be considered in the future. 

Typically, DC fast chargers consists of several power modules. Each power module consists of two power conversion stages (i.e., AC/DC stage and DC/DC), which is presented in Ref. [1]. For power quality analysis, only the impedance of the AC/DC converter is needed since the DC/DC converter is decoupled by the DC-link capacitor.Below, it shows the typical design of the AC/DC converter a DC fast charger's power module. In this typical design, the L-filter, conventional 2-level voltage-source converter, synchronous-frame PLL, synchronous-frame PI controller, and PI controller are used for the power filter, topology, PLL, current controller, and voltage controller, respectively.
<p align="center">
  <img src="./READMEimg/typical design.PNG" alt="Scenarios Tree" width="80%" href="#"/>
</p>

In the table below, the design that has been considered or to be considered are shown
Topology         | Power filter | PLL     | Current control | Voltage control |
:---             | :---         | :---    | :---            | :---            
2-level VSC      | L-filter     | SRF-PLL | SRF-PI          | PI
Vienna rectifier | LCL-filter   | -       | PR              | -

## Impedance model of DC fast chargers
The impedance model of a DC fast charger is essential for estimating the harmonic emission of the charger when it is connected to a non-ideal grid. To run the illustrative notebooks regarding this part, please refer to the folder *Impedance*.
### Impedance at different charging power
It is noted that a charger's impedance changes with the change of the charging power. Thus, with the help of this tool, the charger's impedance at different charging power can be obtained once the design specifications of the charger are given. For instance, in the figure below, it shows the impedance of a DC fast charger at 30 kW, which is obtained with the analytical model in this tool and simulation in PLECS. Note that the two results matches with each other.
<p align="center">
  <img src="./READMEimg/analyticalVSsimulation.PNG" alt="Scenarios Tree" width="100%" href="#"/>
</p>


### Parameter estimation when design specifications are unknown

## Harmonic current source model of DC fast chargers
TBD
## Online grid impedance estimation method
TBD
## Estimation of a fast-charging station's harmonic emission
## References
[1] L Wang, Z Qin, T Slangen, et al. Grid impact of electric vehicle fast charging stations: Trends, standards, issues and mitigation measures-An overview. *IEEE Open
Journal of Power Electronics*,2021, 2: 56-74. [Publication link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9336258).

