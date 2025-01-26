Soot Models
===========


Soot Source Terms
------------------
Spitfire supports a few methods for computing coefficients of soot growth/oxidation/nucleation/etc. in a flamelet library.
These coefficients can be used in the source terms of transport equations for the number and mass density of soot particles.
This two-equation approach arises from representing soot as a monodisperse population of spherical particles.
The source term for the soot number density (:math:`N_s`, in kmol soot / kg air) involves nucleation and coagulation while the source term for the 
soot mass fraction (:math:`Y_s`) involves nucleation, condensation, surface growth, and oxidation.
These can be generally written for laminar flow as

.. math::
    \dot{S}_{N_s} &= (\mathrm{molar\ nucleation}) - (\mathrm{coagulation}) N_s^{11/6}Y_s^{1/6} \\
    \dot{S}_{Y_s} &= (\mathrm{mass\ nucleation}) + (\mathrm{mass\ condensation})N_s + ((\mathrm{surface\ growth}) - (\mathrm{oxidation}) )N_s^{1/3}Y_s^{2/3}
    :label: soot_eqns

When we compute filtered source terms for turbulent flow, we convolve each tabulated coefficient separately,
then substitute these into the laminar equation along with a filtered :math:`\tilde{Y}_s` and :math:`\tilde{N}_s`.
This can be thought of as assuming delta PDFs for :math:`Y_s` and :math:`N_s` and including them as dimensions in the convolutions with statistical independence from the 
other table dimensions (mixture fraction, stoichiometric dissipation rate, etc.).

Spitfire supports three models for computing these source term coefficients: 
  - Aksit_Moss
  - Aksit_Moss_Benzene
  - PAH_alpha

The coefficients are added to a library, given a chemical mechanism and soot model name, through the function ``tabulate_soot_source_coefficients``.

Aksit Moss models
++++++++++++++++++++++++++++

The Aksit Moss models [AksitMoss]_ use acetylene-based nucleation and surface growth rates with empirically-determined parameters.
Only slight modifications are made to the original model when the effect of benzene is included.

.. [AksitMoss] J.B. Moss and I.M. Aksit,
    Modelling soot formation in a laminar diffusion flame burning a surrogate kerosene fuel,
    Proceedings of the Combustion Institute,
    Volume 31,
    2007

In both Aksit-Moss models, the coagulation coefficient (coag) is given by

.. math::
    \mathrm{coag} = \sqrt{\frac{24 V T R_u  N_a}{\rho_s}}\rho^2
    :label: coag_eqn

where :math:`\rho_s` is the density of soot (1800 kg/kmol) and :math:`V` is the volume factor

.. math::
    V = \left(\frac{6}{\pi \rho_s N_a}\right)^{1/3}
    :label: V_eqn

Thermodynamic instability of PAH, along with oxidation below temperatures of about 2000 K, leads to the phenomena referred to as the soot bell curve.
We use a correction factor :math:`B(T)` applied multiplicatively to the nucleation and surface growth rates to account for this.
This bell curve multiplier is written as

.. math::
    B(T) &= \mathrm{max}\left(1 - \exp{\left( \left(\frac{\Delta T_\mathrm{act}}{T_c} - \frac{\Delta T_\mathrm{act}}{T}\right)\right)}, 0\right) \\
    \Delta T_\mathrm{act} &= 16000\ \mathrm{K} \\
    T_c &= 2050\ \mathrm{K}
    :label: B_eqn

For the Aksit_Moss model without benzene, molar nucleation (nucl) and surface growth (surf) depend solely on the concentration of acetylene (:math:`\mathrm{C}_2\mathrm{H}_2`)
with a linear relationship between the number of active sites and the soot surface area:

.. math::
    \mathrm{nucl_{\mathrm{C}_2\mathrm{H}_2}} &= B(T) k_\mathrm{nucl}[\mathrm{C}_2\mathrm{H}_2] \\
    k_\mathrm{nucl} &= 54 \exp{\left(\frac{-21100}{T}\right)}
    :label: nucl_eqn

.. math::
    \mathrm{surf_{\mathrm{C}_2\mathrm{H}_2}} &= \rho A B(T) k_\mathrm{\mathrm{C}_2\mathrm{H}_2}[\mathrm{C}_2\mathrm{H}_2] \\
    k_\mathrm{\mathrm{C}_2\mathrm{H}_2} &= 11700 \exp{\left(\frac{-12000}{T}\right)}
    :label: c2h2_eqn

with an area factor

.. math::
    A = V^2 \pi N_a
    :label: A_eqn

The mass of an incipient particle is given directly as :math:`M_p =144` kg/kmol,
which multiplies the molar :math:`\mathrm{nucl_{\mathrm{C}_2\mathrm{H}_2}}` to get the mass nucleation rate.

The Aksit_Moss_Benzene model uses a nucleation coefficient that is twice that of Aksit_Moss.
The surface growth in Aksit_Moss_Benzene depends on both acetylene and benzene (A1).
The coefficient for acetylene growth is identical to Aksit_Moss while the coefficient for benzene growth is given by

.. math::
    \mathrm{surf}_{\mathrm{A}1} = 20 \rho A B(T) k_\mathrm{\mathrm{C}_2\mathrm{H}_2}[\mathrm{A}1]
    :label: a1_eqn

In both Aksit-Moss models, oxidation depends on the concentrations of both oxygen (:math:`\mathrm{O}_2`) and the hydroxyl radical (:math:`\mathrm{OH}`):

.. math::
    \mathrm{surf}_{\mathrm{O}_2} &= \rho A k_{\mathrm{O}_2} [\mathrm{O}_2] \\
    k_{\mathrm{O}_2} &= 500 \sqrt{T} \exp{\left(\frac{-20000}{T}\right)} \\
    \mathrm{surf}_{\mathrm{OH}} &= \rho A k_{\mathrm{OH}} [\mathrm{OH}] \\
    k_{\mathrm{OH}} &= 4.2325 \sqrt{T}
    :label: ox_eqn


Newer PAH-based models
++++++++++++++++++++++++++++

The PAH_alpha model involves the effects of larger aromatics on the production of soot.

Gas phase molecular collisions are modeled for the free molecular regime,
with self-collisions between molecules of species i having frequency

.. math::
    \beta_{i-i} = 2.2 \sqrt{ \frac{4 \pi R_u T }{ M_C N_{C,i}}} d_p^2
    :label: beta_pah

where :math:`M_C` is the molecular weight of carbon,
:math:`N_{C,i}` is the number of carbon atoms in PAH species i,
and :math:`d_p` is the average diameter of the PAH particles:

.. math::
    d_p =  \left(6 \frac{2 \overline{MW_{\mathrm{PAH}}}}{N_a \pi \rho_s} \right)^{1/3}
    :label: dp_pah

using the density of soot :math:`\rho_s` and average molecular weight (MW) of the PAH species.

Mass nucleation is given by the sum over PAH species :math:`i`:

.. math::
    \mathrm{dimer}_\mathrm{src} = \sum_i 2 MW_i (\gamma_i \beta_{i-i} [\mathrm{PAH}_i]^2 N_a)
    :label: dimer_pah

for the following species with their sticking coefficients :math:`\gamma_i` in parenthesis:
naphthalene(0.001), acenaphthalene (0.003), biphenyl (0.0085), phenanthrene (0.015), acephenanthrylene (0.025), pyrene (0.025), flouranthene (0.025), and cyclopentapyrene (0.039).
Molar nucleation is then given by :math:`\frac{0.5}{361.4} \mathrm{dimer}_\mathrm{src}`.
We note that these nucleation rates tend to dominate and over-predict soot production.
Therefore, when evaluating the source terms with ``compute_soot_source_terms`` a factor of :math:`10^{-6}` is applied to both nucleation rates.

Coagulation is the same as in the Aksit Moss models.

The surface area factor for the surface growth and oxidation rates is

.. math::
    G_A = \rho N_A V^2
    :label: area_pah

A more fundamental approach for modeling surface growth, compared to the Aksit-Moss implementation,
employs elementary reactions to describe the surface chemistry, along with a quasi-steady concentration of radical surface sites.
Following the formulation from [pahsg]_, the surface growth rate for our PAH model is

.. math::
    \mathrm{sg} &= 1.7 \cdot 10^{19} k_4 f^* [\mathrm{C}_2\mathrm{H}_2] \frac{2 M_C}{N_A} G_A \\
    f^* &= \frac{ 1 }{ 1 + \frac{k_{1b} [\mathrm{H}_2\mathrm{O}] + k_{2b} [\mathrm{H}_2] + k_{3b} [\mathrm{H}] + k_4 [\mathrm{C}_2\mathrm{H}_2]}{k_{1f} [\mathrm{OH}] + k_{2f} [\mathrm{H}] + k_{3f}} } \\
    k_{1f} &= 6.72 \cdot 10^{-2} T^{3.33} \exp\left( \frac{-6.09 \cdot 10^6}{R_u T} \right) \\
    k_{1b} &= 6.44 \cdot 10^{-4} T^{3.79} \exp\left( \frac{-27.96 \cdot 10^6}{R_u T} \right) \\
    k_{2f} &= 1.08 \cdot 10^{5} T^{1.8} \exp\left( \frac{-68.42 \cdot 10^6}{R_u T} \right) \\
    k_{2b} &= 8.68 \cdot 10^{1} T^{2.36} \exp\left( \frac{-25.46 \cdot 10^6}{R_u T} \right) \\
    k_{3f} &= 1.13 \cdot 10^{16} T^{-0.06} \exp\left( \frac{-476.05 \cdot 10^6}{R_u T} \right) \\
    k_{3b} &= 4.17 \cdot 10^{10} T^{0.15} \\
    k^4 &= 2.52 \cdot 10^6 T^{1.1} \exp\left( \frac{-17.13 \cdot 10^6}{R_u T} \right)
    :label: sg_pah

where each of the rates :math:`k_i` are in Hz.

.. [pahsg] M. Mueller, G. Blanquart, and H. Pitsch,
    Hybrid method of moments for modeling soot formation and growth,
    Combustion and Flame,
    Volume 156,
    2009

The oxidation rates depend on the concentrations of O, OH, and :math:`\mathrm{O}_2` with linear dependence on soot surface area.
The rate for oxygen has a modified Arrhenius form while
the rates for OH and O are based on Brownian collision of a radical species with a large soot particle.
In the equations below, :math:`M_{\mathrm{O}_2}` is the molecular weight of oxygen,
:math:`M_{\mathrm{OH}}` is the molecular weight of OH, and :math:`M_\mathrm{O}` is the molecular weight of O.

.. math::
    \mathrm{ox}_{\mathrm{O}_2} &= 1.7 \cdot 10^{19} k_{\mathrm{O}_2} \frac{2 M_C}{N_A} [\mathrm{O}_2] G_A \\
    k_{\mathrm{O}_2} &= 2.2 \cdot 10^6 \exp \left( \frac{-31.38 \cdot 10^6}{R_u T} \right)
    :label: o2_pah

.. math::
    \mathrm{ox}_{\mathrm{OH}} = 0.04 M_C \sqrt{\left( \frac{R_u T}{2 \pi M_{OH}} \right)} [\mathrm{OH}] G_A
    :label: oh_pah

.. math::
    \mathrm{ox}_{\mathrm{O}} = 0.2 M_C \sqrt{\left( \frac{R_u T}{2 \pi M_O} \right)} [\mathrm{O}] G_A
    :label: o_pah


Correcting for soot leakage
++++++++++++++++++++++++++++

Soot particles are formed at fuel-rich mixtures and then oxidized as they move toward fuel-lean mixtures.
Many soot models assume infinitely fast oxidation, which means soot is never present near the oxidizer stream.
However, in reality, unoxidized soot can leak into lean mixtures, such as in smoking flames.
One manner in which this can happen is when radiative heat losses are large enough to substantially lower oxidation rates through the lower temperatures,
such as happens in flames with large residence times and Damköhler numbers.
Local extinction events, such as through fast turbulent transport, can also lead to suppressed oxidation.
Finite-rate oxidation models are needed to account for these kinds of phenomena.

Subfilter PDF models have been proposed to capture finite-rate oxidation of soot particles, such as by Colmán et. al. [Col2023]_.
This approach starts by finding the mixture fraction value, termed :math:`Z_\mathrm{soot}`, at which the soot surface growth and soot oxidation rates are equivalent.
A Heaviside function centered at :math:`Z_\mathrm{soot}` represents infinitely fast oxidation, where no soot is present for :math:`Z<Z_\mathrm{soot}`.
Using something like a hyperbolic tangent function smooths this transition and starts to allow for finite-rate oxidation or the presence of soot at :math:`Z<Z_\mathrm{soot}`.

.. [Col2023] Hernando Maldonado Colmán, Antonio Attili, and Michael E. Mueller,
    Large Eddy Simulation of turbulent nonpremixed sooting flames: Presumed subfilter PDF model for finite-rate oxidation of soot,
    Combustion and Flame,
    Volume 258,
    2023

Spitfire supports soot leakage when tabulating soot source term coefficients through the ``percent_soot_leakage`` argument of ``tabulate_soot_source_coefficients``.
When this argument is nonzero, a soot subfilter is defined as

.. math::
    C_s(Z) &= 0.5(1+\mathrm{percent\_soot\_leakage}) + 0.5(1-\mathrm{percent\_soot\_leakage})\tanh((Z - Z_\mathrm{soot} + \zeta)/\zeta) \\
    \zeta &= \frac{\chi (Z_\mathrm{soot})}{2 k_\mathrm{ox}(Z_\mathrm{soot})}
    :label: corr_eqn

where :math:`k_\mathrm{ox}` is the sum of oxidation coefficients which we interpolate over mixture fraction and evaluate at :math:`Z_\mathrm{soot}`.
This soot subfilter then multiplies all surface growth and oxidation coefficients in the library.
The soot subfilter is also added to the library as the property 'soot_subfilter'.
When building a turbulent library, the corrected surface growth and oxidation coefficients (:math:`\mathrm{surf}_i C_s`) are convolved with the mixture fraction PDF
and normalized by the convolution of the soot subfilter itself with that same PDF:

.. math::
    \widetilde{\mathrm{surf}_i} = \frac{\int \mathrm{surf}_i C_s P_z dZ}{\int C_s P_z dZ}
    :label: turb_corr_eqn


Soot Radiation Models
----------------------
Absorption and radiation soot coefficients can be added onto a library through the function ``tabulate_soot_radiation_model``.
Spitfire currently supports a 'linear_absorption_coefficient' model for soot radiation and absorption.

The absorption coefficient is computed as

.. math::
    \mathrm{soot\_abs\_coeff} = \mathrm{max}(\mathrm{intercept} + \mathrm{slope} * T, \mathrm{min\_value}) \frac{\rho}{\rho_s}
    :label: soot_abs_coeff

where the parameters 'intercept', 'slope' and 'min\_value' are provided by the user as desired.
Default values for all parameters are zero, which would result in zero absorption/radiation if left unchanged.
Suggested values to feed the function are ``input_modelargs = {'slope': 1735., 'intercept': -3.75e5}``.

The radiation source is then related to the absorption as

.. math::
    \mathrm{soot\_rad\_source} = \mathrm{soot\_abs\_coeff} \sigma_B T^4 / \pi \ \ (\mathrm{MW}/\mathrm{m}^3)
    :label: soot_rad_source


Gas Radiation Models
---------------------
Absorption and radiation gas coefficients can be added onto a library through the function ``tabulate_gas_radiation_model``.
Spitfire supports a 'Barlow_GreyGas' model for gas radiation and absorption.

The absorption coefficient is computed as

.. math::
    \mathrm{gas\_abs\_coeff} = \frac{R_u}{101325.} \rho T \sum_i (\mathrm{abs}_i(T) y_i/MW_i) \ \ (\mathrm{m}^{-1})
    :label: gas_abs_coeff

for the sum over the species :math:`\mathrm{CH}_4`, :math:`\mathrm{H}_2\mathrm{O}`, :math:`\mathrm{CO}`, :math:`\mathrm{CO}_2` absorption fits (if present in the chemical mechanism) given by

.. math::
    \mathrm{abs}_{\mathrm{CH}_4}(T) &= c_0 + c_1 T + c_2 T^2 + c_3 T^3 + c_4 T^4 \\
    c_0 &= 6.6334 \\
    c_1 &= -0.0035686\\
    c_2 &= 1.6682\cdot 10^{-8}\\
    c_3 &= 2.5611 \cdot 10^{-10}\\
    c_4 &= -2.6558 \cdot 10^{-14}
    :label: abs_ch4

.. math::
    \mathrm{abs}_{\mathrm{H}_2\mathrm{O}}(T) &= c_0 + \frac{1000}{T} (c_1 + \frac{1000}{T} (c_2 + \frac{1000}{T} (c_3 + \frac{1000}{T} (c_4 + \frac{1000}{T} + c_5)))) \\
    c_0 &= -0.23093\\
    c_1 &= -1.12390\\
    c_2 &= 9.41530\\
    c_3 &= -2.99880\\
    c_4 &= 0.51382\\
    c_5 &= -1.86840 \cdot 10^{-5}
    :label: abs_h2o
    
.. math::
    \mathrm{abs}_{\mathrm{CO}_2}(T) &= c_0 + \frac{1000}{T} (c_1 + \frac{1000}{T} (c_2 + \frac{1000}{T} (c_3 + \frac{1000}{T} (c_4 + \frac{1000}{T} + c_5)))) \\
    c_0 &= 18.741\\
    c_1 &= -121.310\\
    c_2 &= 273.500\\
    c_3 &= -194.050\\
    c_4 &= 56.310\\
    c_5 &= -5.8169
    :label: abs_co2

.. math::
    \mathrm{abs}_{\mathrm{CO}} &= c_0 + T (c_1 + T (c_2 + T (c_3 + T c_4))) \\
    c_0 &= \begin{cases}
            10.09 \ \ \mathrm{if} \ \ T \geq 750 \\
            4.7869 \ \ \mathrm{otherwise}
           \end{cases} \\
    c_1 &= \begin{cases}
            -0.01183 \ \ \mathrm{if} \ \ T \geq 750 \\
            -0.06953 \ \ \mathrm{otherwise}
           \end{cases} \\
    c_2 &= \begin{cases}
            4.7753 \cdot 10^{-6} \ \ \mathrm{if} \ \ T \geq 750 \\
            2.95775\cdot 10^{-4} \ \ \mathrm{otherwise}
           \end{cases} \\
    c_3 &= \begin{cases}
            -5.87209 \cdot 10^{-10} \ \ \mathrm{if} \ \ T \geq 750 \\
            -4.25732\cdot 10^{-7} \ \ \mathrm{otherwise}
           \end{cases} \\
    c_4 &= \begin{cases}
            -2.5334 \cdot 10^{-14} \ \ \mathrm{if} \ \ T \geq 750 \\
            2.02894\cdot 10^{-10} \ \ \mathrm{otherwise}
           \end{cases}
    :label: abs_co

The radiation source is then related to the absorption as

.. math::
    \mathrm{gas\_rad\_source} = \mathrm{gas\_abs\_coeff} \sigma_B T^4 / \pi \ \ (\mathrm{MW}/\mathrm{m}^3)
    :label: gas_rad_source


Tutorials
---------

.. toctree::
    :maxdepth: 1

    demo/flamelet/soot_modeling

