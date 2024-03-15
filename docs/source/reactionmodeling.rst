Reaction Models
===============

.. toctree::
    :maxdepth: 1
    :caption: Demonstrations:

    demo/reactors/thermochemistry_Cantera_Spitfire_griffon
    demo/reactors/one_step_heptane_ignition
    demo/reactors/ignition_delay_NTC_DME
    demo/reactors/oscillating_ignition_extinction
    demo/reactors/isothermal_reactors_with_mode_analysis
    demo/reactors/ignition_extinction_heptane
    demo/flamelet/introduction_to_flamelets
    demo/flamelet/high_level_tabulation_api_adiabatic
    demo/flamelet/example_nonadiabatic_flamelets
    demo/flamelet/methane_shear_layer_tabulation
    demo/flamelet/transient_ignition_stepper_details
    demo/flamelet/example_transient_flamelet_rate_sensitivity
    demo/flamelet/example_coal_combustion_model



The following sections detail the backround and some of the math behind Spitfire's reaction modeling capabilities.
We present the background of non-premixed flamelet models and then of simpler homogeneous reactor systems,
and supported reaction rate laws and thermodynamic property models.

Flamelet Modeling Background
++++++++++++++++++++++++++++

**Mixture Fraction Models**

The basic starting point for modeling reacting flows is direct numerical simulation (DNS),
an approach wherein governing equations for mass, momentum, energy, and species transport are solved
on fine grids with small time steps to simply resolve every scale present in the problem.
The range of time and length scales as well as state space dimensionality
make DNS impossible for engineering analysis of turbulent flows with current computing capabilities.
This motivates turbulence models that require closure of thermochemical state evolution.
For non-premixed flames an effective approach is RANS or LES with mixture fraction modeling.
Here we describe mixture fraction models for low-Mach combustion applications, particularly large pool fires,
and we avoid getting into the details of RANS vs LES models.
See Domino et al. [DHKH20201]_ for a more detailed overview, particularly focused on capabilities and applications at Sandia National Laboratories (SNL).

.. [DHKH20201] Stefan P. Domino and John C. Hewson and Robert C. Knaus and Michael A Hansen,
    Predicting large-scale pool fire dynamics using an unsteady flamelet- and large-eddy simulation-based model suite,
    Physics of Fluids,
    Volume 33,
    2021

In mixture fraction models the standard set of filtered continuity and momentum equations are solved
along with any equations needed by the turbulence model (*e.g.*, a :math:`k-\varepsilon` model requires two additional equations).
For large, sooting fires radiation plays a significant role and must be modeled, typically through a discrete ordinates approach.
The remaining ingredient to a mixture fraction model is the evolution of the thermochemical state.
Solving transport equations for a detailed set of chemical species is impractical 
due to the large numbers of species and reactions
and the difficulty of accurately predicting subgrid fuel consumption.
Mixture fraction models simultaneously address the issues of state space dimensionality and turbulent closure,
separating the small-scale resolution of detailed chemical reactions from the large-scale evolution of a fire.

All mixture fraction models solve at least one additional transport equation for the mixture fraction, a conserved scalar that measures fuel/air mixing.
RANS models include at least one more equation for the subgrid variance of the mixture fraction (the *scalar variance*),
while LES models typically presume a gradient closure to measure subgrid variance.

**Equilibrium & Burke-Schumann Models**

Presuming that combustion chemistry happens infinitely fast leads to equilibrium or Burke-Schumann models,
wherein the mixture fraction and its variance completely describe the thermochemical state.
These models are distinguished by their treatment of chemistry - equilibrium corresponds to minimization of the Gibbs free energy while Burke-Schumann presumes idealized combustion.
For a laminar flame (the limit of zero subgrid scalar variance) this means the temperature and mass fractions are known from the local mixture fraction alone.
This is an extreme assumption on scale separation - presuming that chemistry responds so fast that it may be completely decoupled from the fluid dynamics.

**Strained Laminar Flamelet Models**

Improved mixture fraction models incorporate the effects of local fluid strain on combustion chemistry.
A useful measure of the strain rate is the scalar dissipation rate, :math:`\chi=2D\|\nabla\mathcal{Z}\|^2`, where :math:`D` is the diffusivity of the mixture fraction (typically near the thermal diffusivity).
In these models gas state variables and properties are known from the mixture fraction, scalar variance, and scalar dissipation rate,
with the dissipation rate computed in a variety of ways in LES/RANS approaches.
At high enough strain rates flames can locally extinguish although this is not typically included in these models.
While this type of model is still assuming scale separation, the effects of finite rate chemistry are included through a diffusive-reactive balance described by the flamelet equations parameterized by the scalar dissipation rate.
This is referred to as a strained laminar flamelet model, or SLFM.
An essential assumption to SLFM is that to solve the flamelet equations of an SLFM method one must presume a correlation of the scalar dissipation rate and mixture fraction, :math:`\chi(\mathcal{Z})`.
Spitfire's preliminary aim related to the solution of the reaction-diffusion equations needed to build SLFM chemistry libraries.


**Non-adiabatic Flamelet Models**

An additional advancement of mixture fraction models is necessary for application to large-scale sooting fires: non-adiabaticity of the flame.
Radiative losses to soot and exposed surfaces removes the correlation of mixture fraction and enthalpy that holds for adiabatic flows.
In non-adiabatic mixture fraction models the fire model must determine the extent of departure from adiabatic conditions,
and the flamelet model must account for the effect of this departure on small-scale combustion chemistry.
Similarly to the dissipation rate of SLFM models, the effect of radiation on the flamelet equations must be modeled in mixture fraction space.
Alternatively one may alter boundary stream enthalpies.
Equation modifications typically take the form of temperature-dependent source terms to model radiative quenching/extinction,
although superadiabaticity in extremely large fires may be important as well.
Two approaches to equation modification are supported in Spitfire and will be presented in detail in a following section.
Non-adiabatic equilibrium and Burke-Schumann flamelets may also be constructed with Spitfire, and this will also be presented later.

**Turbulence Modeling: Presumed PDF Tabulation**

An important step for RANS/LES calculations is the closure problem: modeling the effects of subgrid (unresolved) phenomena on resolved scales.
With presumed PDF models, one can view the various flamelet models discussed above as "chemistry models" or "reaction models"
and the problem of subgrid closure requires a follow-on "mixing model."
Simple presumed PDF models require convolution of tabulated flamelet solutions over the mixture fraction dimension for a range of scalar variances.
This is discussed in two demonstrations, :ref:`Tabulation API Example: Presumed PDF SLFM Tables` and :ref:`Custom Presumed PDF: log-mean PDF of the scalar dissipation rate`.


**Soot Modeling**

One final step to the solution of sooting fires is the modeling of soot,
which is often done with a moment method or sectional method.
An efficient, semi-empirical method appropriate for engineering analysis is a two-equation soot model
where transport equations for the number and mass density of soot particles are solved.
Such a model requires source terms for soot formation/destruction based on the local thermochemical state.
Physical processes such as particle coagulation are resolved by the fire model,
but chemical processes such as gas phase precursor formulation, nucleation, condensation, surface growth and oxidation depend on the flamelet model for the chemistry.
Soot moments are solved in the fire model because soot particles evolve relatively slowly and their effective diffusion rates are low compared to gas species.
It is inappropriate to presume that soot particles follow the same diffusive-reactive equilibrium as gas species,
and so we compute coefficients, or propensity terms, for soot growth/oxidation/nucleation/etc. in a flamelet library.

**Tabulation**

The various flamelet formulations described above can be prepared in advance and *tabulated* so that equilibrium/flamelet solutions
do not need to be computed *in situ* during a simulation.
Spitfire currently provides several layers of abstraction for building flamelet libraries in the form of structured interpolation tables,
and is actively involved in research into unstructured tabulation approaches (*e.g.*, neural networks that do not employ tensor product interpolation).
Direct support is available for tables used at Sandia National Laboratories for hydrocarbon pool fire models,
although Spitfire is designed to allow advanced users to tabulate libraries for their own formulations.




Governing Equations for Non-premixed Flamelets
++++++++++++++++++++++++++++++++++++++++++++++

Spitfire's flamelet and tabulation capabilities support generalized flamelet models for RANS/LES of hydrocarbon pool fires in the non-premixed combustion regime.
While direct support is provided for sooting hydrocarbon fires,
Spitfire leverages a modular, extensible stack from thermochemistry evaluator, numerical methods, solvers, and tabulation techniques.
In this section we detail the governing equations solved by Spitfire for equilibrium, strained steady/transient, adiabatic/non-adiabatic flamelets.
All of these equations are defined with the mixture fraction coordinate.
These equations can be derived in a variety of ways, either through local coordinate transformations along the isosurfaces of the mixture fraction
or by transforming species transport equations to follow a general Lagrangian coordinate frame.
In both cases, time and length scale separation arguments lead to one-dimensional equilibrium/strained/non-adiabatic flamelet models of flows with two inlets.

**Strained Flamelet Models**

The thermochemical state is described with the mass fractions :math:`Y_i` and temperature :math:`T`.
These are described by the mixture fraction :math:`\mathcal{Z}` and other parameters, and in transient flamelets we consider
evolution in a Lagrangian time :math:`t`.
Equations :eq:`adiabatic_flamelet_Yi_eqn` and :eq:`adiabatic_flamelet_T_eqn` describe adiabatic flamelets,
which balance molecular mixing (with strength proportional the scalar dissipation rate :math:`\chi`) with species production/consumption and heat release through chemical reactions.
These equations include variable heat capacity effects and the full form of the heat flux including the diffusive enthalpy terms.
They do not account for differential diffusion and are based on a unity Lewis number and mass fraction form of the species diffusive fluxes.
Generalizations of the diffusive flux models have been derived and are planned for a future version of Spitfire.
The variable heat capacity term and enthalpy flux terms are optional in Spitfire, but are included by default as they are very important for modeling non-adiabatic flows with strong coupling to radiative losses.
The steady version of this equation set is derived by simply removing the time derivative term.

.. math::
    \underbrace{\frac{\partial Y_i}{\partial t}}_{\text{transient}}
    = 
    \underbrace{\frac{\chi}{2}\frac{\partial^2 Y_i}{\partial \mathcal{Z}^2}}_{\text{molecular diffusion}}
    + 
    \underbrace{\frac{\omega_i}{\rho}}_{\text{chemistry}},
    :label: adiabatic_flamelet_Yi_eqn

.. math::
    \underbrace{\frac{\partial T}{\partial t}}_{\text{transient}}
    = 
    \underbrace{\frac{\chi}{2}\frac{\partial^2 T}{\partial \mathcal{Z}^2}}_{\text{molecular diffusion}}
    + 
    \underbrace{\frac{\chi}{2}\frac{\partial T}{\partial \mathcal{Z}}\sum_{i=1}^{n}\frac{c_{p,i}}{c_p}\frac{\partial Y_i}{\partial \mathcal{Z}}}_{\text{enthalpy diffusion}}
    + 
    \underbrace{\frac{\chi}{2}\frac{1}{c_p}\frac{\partial c_p}{\partial \mathcal{Z}}\frac{\partial T}{\partial \mathcal{Z}}}_{\text{variable cp term}} 
    - 
    \underbrace{\frac{1}{\rho c_p}\sum_{i=1}^{n}\omega_i h_i}_{\text{chemistry}}.
    :label: adiabatic_flamelet_T_eqn

These equations are supplemented by Dirichlet boundary conditions defined by the oxidizer and fuel states,

.. math::
    T(t, 0) &= T_{\mathrm{oxy}}, \\
    Y_i(t, 0) &= Y_{i,\mathrm{oxy}}, \\
    T(t, 1) &= T_{\mathrm{fuel}}, \\
    Y_i(t, 1) &= Y_{i,\mathrm{fuel}}.


The dissipation rate :math:`\chi` can be a constant or depend on the mixture fraction.
Spitfire provides the common form due to Peters:

.. math::
    \chi(\mathcal{Z}) = \chi_{\mathrm{max}} \exp\left( -2\left[\mathrm{erfinv}(2\mathcal{Z}-1)\right]^2 \right).

**Non-Adiabatic Strained Flamelet Models**

Spitfire also supports non-adiabatic flamelet models.
Non-adiabaticity modifies the temperature equation through the inclusion of source terms,

.. math::
    \frac{\partial T}{\partial t} = 
    \left.\frac{\partial T}{\partial t}\right|_{\mathrm{adiabatic}} + 
    \underbrace{\frac{h}{\rho c_p}(T_\infty - T)}_{\text{simple convective term}} + 
    \underbrace{\frac{\varepsilon\sigma}{\rho c_p}(T_\mathrm{surf}^4 - T^4)}_{\text{simple radiative term}}.
    :label: non-adiabatic_flamelet_T_eqn

The convection and radiation coefficients and reference temperatures are allowed to vary over the mixture fraction.
Further, the heat loss terms can be scaled to facilitate simulation of strong radiative quenching in large sooting fires,

.. math::
    \frac{\partial T}{\partial t} = 
    \left.\frac{\partial T}{\partial t}\right|_{\mathrm{adiabatic}} + 
    \underbrace{\frac{\chi_{\mathrm{max}}}{\mathcal{Z}_{\mathrm{st}}(1 - \mathcal{Z}_{\mathrm{st}})}\frac{h}{\rho c_p}\frac{T_\infty - T}{T_{\mathrm{max}} - T_\infty}}_{\text{scaled convective term}} +
    \underbrace{\frac{\chi_{\mathrm{max}}}{\mathcal{Z}_{\mathrm{st}}(1 - \mathcal{Z}_{\mathrm{st}})}\frac{\varepsilon\sigma}{\rho c_p}\frac{T_\mathrm{surf}^4 - T^4}{T_{\mathrm{max}}^4 - T_\mathrm{surf}^4}}_{\text{scaled radiative term}}.
    :label: scaled_non-adiabatic_flamelet_T_eqn

where :math:`\mathcal{Z}_{\mathrm{st}}` is the stoichiometric mixture fraction.
The preferred model supported in Spitfire for hydrocarbon pool fires is to use a convective/linear form of the heat loss with :math:`h'` to :math:`10^6-10^7`,
which rapidly drives hydrocarbon flamelets to extinction, covering a wide range of the heat loss space broadly throughout the mixture fraction.
Spitfire also provides a quasi-steady continuation approach for non-adiabatic SLFM models.

**Equilibrium Models**

The previous section provides unsteady flamelet equations for adiabatic and non-adiabatic flames.
Steady flamelet models for SLFM tabulation are obtained by neglecting the time derivative terms.
Equilibrium models are derived by neglecting the dissipation terms as well.
Here the temperature equation can be replaced with a constant total enthalpy equation (:math:`h_t=0`),
which is a more basic expression than the temperature equations above.
Minimizing the Gibbs free energy at each grid point in :math:`\mathcal{Z}` then produces the desired equilibrium solution.
Spitfire does not provide an equilibrium solver, instead using the well-proven capability of Cantera for equilirium calculations.


**Burke-Schumann Models**

Burke-Schumann flamelet models are based on a presumption of infinitely fast and *idealized* combustion,
wherein fuel and oxygen are consumed to produce water and carbon dioxide products,
to the extent afforded by local stoichiometry.
At the stoichiometric mixture fraction this results in complete combustion.
The enthalpy of the mixture is kept at its linear value between the fuel/oxidizer streams,
and mass fractions are computed with the ideal combustion assumption.
The temperature is then computed given the enthalpy and composition,
often resulting in sharp peaks at :math:`\mathcal{Z}_{\mathrm{st}}` and nearly linear profiles on the lean and rich sides.



**Non-Adiabatic Equilibrium and Burke-Schumann Models**

The non-strained flamelet models can be augmented to account for non-adiabaticity in a number of ways.
In Spitfire we implement a fairly simple formulation.
A piecewise linear function of mixture fraction, zero at the boundaries and unity at :math:`\mathcal{Z}_{\mathrm{st}}`,
is multiplied by a (negative) stoichiometric enthalpy defect parameter to offset the enthalpy of the mixture,
and then one simply recomputes equilibrium/Burke-Schumann profiles with this perturbed enthalpy.
This is performed for enthalpies spanning the nonreacted enthalpy and the enthalpy of a completely extinguished mixture
to model the full extent of radiative quenching/extinction.

An intriguing and untested alternative more consistent with the SLFM approach is to introduce a temperature source term.
This results in a collection of transient homogeneous reactor problems,
which could be solved in tandem and tabulated as if they were flamelet equations with zero strain.
This may lead to a very different quenching behavior than the imposed enthalpy defect profiles,
although this is likely not worth pursuing simply due to the limitations of equilibrium/Burke-Schumann models.




Tabulation Approach for Turbulent Pool Fires
++++++++++++++++++++++++++++++++++++++++++++


The ``FlameletSpec`` Class
""""""""""""""""""""""""""
The first step of building flamelet libraries is to specify the thermochemistry, mixture fraction grid, boundary streams, and flamelet equation set.
Spitfire provides the ``FlameletSpec`` class to contain this information, which is detailed below.
``FlameletSpec`` instances can be created by specifying these parameters directly or 
by providing a one-dimensional slice of a ``Library`` object with a ``mech_spec`` object as an ``extra_attribute``.
The library option is extremely convenient when building multidimensional flamelet libraries.



- thermochemistry:
    - ``mech_spec``: a ``Spitfire.ChemicalMechanismSpec`` object representing gas thermochemistry (species thermal data, transport data, reaction rate data)
    - ``initial_condition``: a string (see :ref:`Building Adiabatic Non-Strained Libraries`) or `numpy.ndarray` for the interior state variables (a `ravel`ed grid with temperature and :math:`n-1` mass fractions for :math:`n_{\mathcal{Z}}-2` grid points)

- discretization of the mixture fraction:
    - ``grid``: directly specify a set of grid point locations, overriding any other parameters
    - ``grid_points``: specify the number of grid points
    - ``grid_type``: specify to use a "clustered" grid or a "uniform" grid
    - ``grid_cluster_point``: mixture fraction value around which a "clustered" grid refines the mesh, can be "stoichiometric" to cluster around :math:`\mathcal{Z}_{\mathrm{st}}`
    - ``grid_cluster_intensity``: how tightly a "clustered" grid reduces the grid spacing around its cluster point

- boundary streams:
    - ``oxy_stream``: Cantera ``Quantity`` or ``Solution`` instance for the oxidizer (:math:`\mathcal{Z}=0`)
    - ``fuel_stream``: Cantera ``Quantity`` or ``Solution`` instance for the fuel (:math:`\mathcal{Z}=1`)
  
- equation terms:
    - ``include_enthalpy_flux``: whether or not to include the enthalpy diffusion term, see :eq:`adiabatic_flamelet_T_eqn`
    - ``include_variable_cp``: whether or not to include the variable heat capacity term (heat capacities are always modeled as specified in the chemical mechanism), see :eq:`adiabatic_flamelet_T_eqn`
    - heat loss formulation
        - ``heat_transfer``: whether the flamelets are "adiabatic" or "non-adiabatic"
        - ``convection_temperature``: directly specify a constant or :math:`\mathcal{Z}`-dependent value for :math:`T_\infty`
        - ``radiation_temperature``: directly specify a constant or :math:`\mathcal{Z}`-dependent value for :math:`T_{\mathrm{surf}}`
        - ``convection_coefficient``: directly specify a constant or :math:`\mathcal{Z}`-dependent value for :math:`h`
        - ``radiative_emissivity``: directly specify a constant or :math:`\mathcal{Z}`-dependent value for :math:`\varepsilon`
        - ``scale_heat_loss_by_temp_range``: whether or not to include the :math:`T_\max` terms in :eq:`scaled_non-adiabatic_flamelet_T_eqn`
        - ``scale_convection_by_dissipation``: whether or not to multiply both heat loss terms by :math:`\chi_\max` in :eq:`scaled_non-adiabatic_flamelet_T_eqn`
        - ``use_linear_ref_temp_profile``: specifies a linear temperature profile for :math:`T_\infty` and :math:`T_{\mathrm{surf}}` based on the stream temperatures
    - dissipation rate formulation
        - ``dissipation_rate``: option to specify the dissipation rate evaluated on the mixture fraction grid, ignores all other dissipation rate options
        - ``dissipation_rate_form``: whether to use a "Peters" or a constant dissipation rate form, based on the given maximum or stoichiometric values
        - ``max_dissipation_rate``: if using the "Peters" or constant dissipation rate, calibrate the curve to hit this maximum value
        - ``stoich_dissipation_rate``: if using the "Peters" or constant dissipation rate, calibrate the curve to hit this value at the stoichiometric mixture fraction


The ``Flamelet`` Class
""""""""""""""""""""""
After building an instance of the ``FlameletSpec`` class,
we build a ``Flamelet`` instance.
The ``Flamelet`` class provides basic interfaces for the flamelet equations,
namely initializing to equilibrium and Burke-Schumann solutions (see :ref:`Building Adiabatic Non-Strained Libraries`) 
and computing steady/transient solutions.

To compute steady solutions, the ``Flamelet`` class provides the ``compute_steady_state`` method,
which will use the initial condition of the flamelet as a first guess and attempt to compute the steady flamelet solution with a variety of solvers.
Constructing SLFM tables can often be done by chaining together calls to ``compute_steady_state`` while traversing a parameter space.

For transient flamelets, the ``integrate`` method wraps a call to Spitfire's ``odesolve`` function,
and methods like ``integrate_to_steady``, ``integrate_to_time``, ``integrate_to_steady_after_ignition``, ``integrate_for_heat_loss``, and ``compute_ignition_delay``
further provide simpler interfaces around ``integrate``.

Residuals and Jacobian matrices are computed with code in Spitfire's C++ engine, Griffon,
and abstract numerical algorithms like Newton's method and time integration are coded directly in Python.
A specialized version of Newton's method exists for flamelets as well as a novel pseudotransient continuation approach (CITE MJ SIAM/CTM).
The ``Flamelet`` class provides the ``rhs`` and ``jac`` methods so a user may directly write their own solvers or use external time integrators.
``jac`` provides the Jacobian in a form suitable for a particular linear solver,
and the ``jac_csc`` method converts this to a SciPy CSC format.
Spitfire can be made to use the SuperLU solver provided through SciPy, but Spitfire's built-in solver is signficantly faster
because it is explicitly designed for the "block-tridiagonal-diagonal-offdiagonal matrices" we build for flamelets.




Building Adiabatic Non-Strained Libraries
"""""""""""""""""""""""""""""""""""""""""
Adiabatic, non-strained flamelet libraries are the simplest to construct.
Nonreacting, equilibrium, and Burke-Schumann flamelets can be obtained simply by initializing a ``Flamelet`` instance
with the appropriate ``initial_condition`` argument.
Flamelets may also be initialized to linear temperature and mass fractions through the ``FlameletSpec`` directly,
as an alternative to the nonreacting flamelets with linear enthalpy and mass fractions.

Several wrapper methods, termed "library builder" methods, are provided for convenience:
- ``build_unreacted_library``: ``FlameletSpec(..., initial_condition='unreacted', ...)``
- ``build_adiabatic_eq_library``: ``FlameletSpec(..., initial_condition='equilibrium', ...)``
- ``build_adiabatic_bs_library``: ``FlameletSpec(..., initial_condition='Burke-Schumann', ...)``
- no ``build_*_library`` method for extinguished states, use ``FlameletSpec(..., initial_condition='linear-TY', ...)``

The ``build_*_library`` methods above are quite simple,
copying the provided ``FlameletSpec`` instance,
setting the ``initial_condition``, 
creating a ``Flamelet`` instance and then returning the desired ``Library``.
For instance, a simple implementation of the equilibrium library is shown here::
    
    def build_adiabatic_eq_library(flamelet_specs, verbose=True):
        # 1) copy flamelet specs, make FlameletSpecs if a dictionary is provided
        # 2) set the initial condition
        # 3) instantiate a ``Flamelet``
        # 4) return a ``Library`` object

        fs = FlameletSpec(**flamelet_specs) if isinstance(flamelet_specs, dict) else copy.copy(flamelet_specs)
        fs.initial_condition = 'equilibrium'
        flamelet = Flamelet(fs)
        return flamelet.make_library_from_interior_state(flamelet.initial_interior_state)


Building Adiabatic SLFM Libraries
"""""""""""""""""""""""""""""""""
To build adiabatic SLFM libraries we will Spitfire's ``Flamelet.compute_steady_state()`` method along with one of the initialization approaches (:ref:`Building Adiabatic Non-Strained Libraries`).
This method is demonstrated along with transient flamelets and more by the :ref:`Introduction to Flamelet Models & Spitfire` demonstration.
The basic SLFM library is composed by computing a steady flamelet solution for low values of the scalar dissipation rate,
and then computing steady solutions for increasing :math:`\chi` until reaching a specified value or when extinction occurs.

The ``build_adiabatic_slfm_library`` method manages this simple parameter continuation problem.
Default behavior is to initialize to the equilibrium flamelet first, and then directly solve for a list of specified maximum/stoichiometric dissipation rates.
Advanced users can fairly easily compose controller-type continuation approaches for more accurate (and possibly faster) tabulation,
and generalization of these techniques has long been on Spitfire's to-do list.
The :ref:`Tabulation API Example: Adiabatic Flamelet Models`,
:ref:`Tabulation API Example: Methane Shear Layer`,
and :ref:`Custom Tabulation Example: 4D Coal Combustion Model`
pages demonstrate the use of ``build_adiabatic_slfm_library`` and ``Flamelet.compute_steady_state()``
to generate several steady flamelet libraries.
The :ref:`Custom Tabulation Example: 4D Coal Combustion Model` demonstration showcases a four-dimensional
steady flamelet library with flame strain, heat loss, and variable fuel composition relevant to multiphase combustion.



Building Non-Adiabatic Non-Strained Libraries
"""""""""""""""""""""""""""""""""""""""""""""
To build nonadiabatic variants of the non-strained libraries, Spitfire employs a straightforward approach used at SNL.
The initial adiabatic libraries are created as described above,
and then a range of stoichiometric enthalpy defect values is traversed, recomputing equilibrium composition and temperature,
or just the temperature for Burke-Schumann models, for each enthalpy profile.
The `build_nonadiabatic_defect_eq_library` and `build_nonadiabatic_defect_bs_library` methods that execute this procedure
do not leverage the `Flamelet` class very much beyond initialization.
These tables could be built more easily if we improved `Flamelet` to allow enthalpy-based initialization (instead of temperature).
Examples of non-adiabatic, non-strained flamelet libraries may be found in the :ref:`Tabulation API Example: Nonadiabatic Flamelet Models` demonstration.


Building Non-Adiabatic SLFM Libraries
"""""""""""""""""""""""""""""""""""""
Spitfire provides `build_*_library` methods for two approaches to building non-adiabatic strained laminar flamelet libraries.
Each of these approaches builds a `Library` object with mixture fraction, dissipation rate, and stoichiometric enthalpy defect dimensions.
At the moment neither of these capabilities can model super-adiabatic flamelets (*e.g.*, positive "heat loss").

- `build_nonadiabatic_defect_transient_slfm_library`: transient quenching to a full enthalpy defect range, more uniform :math:`T(\mathcal{Z})` profiles near extinction
- `build_nonadiabatic_defect_steady_slfm_library`: quasi-steady heat loss, often does not extinguishe the mixture or reach very large enthalpy defect values

The transient version of heat loss is used at SNL to model large-scale sooting pool fires
and has been carefully designed to cover a wide range of enthalpy defects (full radiative quenching)
with broad heat loss profiles in mixture fraction.
It relies on running what is ultimately an unstable calculation,
solving :eq:`scaled_non-adiabatic_flamelet_T_eqn` with all of the "corrections" applied to the convective heating term (the :math:`T^4` term is not used despite the apparent relationship to radiative quenching)
until the temperature is dropped to within 5% of the linear profile between the two streams.
This is typically run with the convective heating coefficient at :math:`10^7` to drive extreme heat loss.
The `integrate_for_heat_loss` method on the `Flamelet` class is used to drive this transient calculation,
and often several runs are attempted by Spitfire with lower and lower values of the time integration tolerance (increasing how conservatively we run adaptive time stepping).
This is performed for each steady solution in an adiabatic SLFM table,
and extension of the enthalpy defect for each dissipation rate can be run in parallel.

The quasi-steady version of heat loss uses the `compute_steady_state` method to perform continuation in the convective heat loss parameter.
This is much more similar to typical SLFM tabulation approaches,
but it typically does not result in as large of enthalpy defect ranges.

In both versions of the enthalpy defect extension,
results are interpolated onto a grid of the stoichiometric enthalpy defect.
This step introduces a slight amount of error but is necessary to allow structured (tensor product) interpolation in the downstream CFD code.
Examples of these and non-strained variants may be found in the :ref:`Tabulation API Example: Nonadiabatic Flamelet Models` demonstration.

For some additional demonstrations of transient flamelet libraries,
the :ref:`Introduction to Flamelet Models & Spitfire` demonstration shows a transient adiabatic flamelet model of extinction,
and the :ref:`Transient Flamelet Example: Ignition and Advanced Time Integration` example show transient flamelet ignition calculations with a variety of high-order time integrators.






Governing Equations for Homogeneous Reactors
++++++++++++++++++++++++++++++++++++++++++++
Homogeneous, or 'zero-dimensional,' reactor models represent well-mixed combustion systems wherein there are no spatial gradients in any quantity describing the chemical mixture.
In such a system the temperature :math:`T`, pressure :math:`p`, and composition, expressed by the mass fractions :math:`\{Y_i\}`, are all homogeneous and a reactor may be modeled as a point in space whose properties vary only in time, :math:`t`.
Zero-dimensional systems are idealizations of very complex systems but have their place in the modeling of combustion processes.
In a lab setting this idealization can be approached with *jet-stirred reactors* (JSR, also commonly referred to as a continuous stirred tank reactor (CSTR), perfectly stirred reactor (PSR), and Longwell reactor) and *rapid compression machines* (RCM).
A JSR is a continuous stirred tank reactor to which reactants are fed and mixed rapidly through several opposed jets.
A JSR unit coupled with downstream gas chromatography and mass spectrometry can be used to quantify the composition of the chemical mixture as reaction proceeds.
Detailed models of combustion kinetics are developed through comparison with experimental data from such systems.
The rapid computational solution of kinetic models for simple, zero-dimensional reactors is of great fundamental importance to combustion modeling.

In Spitfire we model mixtures of ideal gases in twelve types of reactors distinguished by their *configuration*, *heat transfer*, and *mass transfer*.
We use *configuration* to distinguish isochoric, or constant-volume, reactors from isobaric, or constant-pressure, ones.
*Mass transfer* refers to a closed, or batch, reactor or an open reactor with mass flow at specified mean residence time.
Three types of *heat transfer* are available:

- adiabatic: a reactor with insulated walls that allow no heat transfer with the surroundings
- isothermal: a reactor whose temperature is held exactly constant for all time
- diathermal: a reactor whose walls allow a finite rate of heat transfer by radiative heat transfer to a nearby surface and convective heat transfer to a fluid flowing around the reactor

Below we detail the equations governing isochoric and isobaric reactors with any pair of models for mass and transfer.
In all cases, the gas is modeled as a mixture of thermally perfect gases.
The ideal gas law applies to each species and the bulk mixture.

.. math::
 p = \rho R_\mathrm{mix} T,
 :label: ideal_gas_law

where the mixture specific gas constant, :math:`R_\mathrm{mix}`, is the universal molar gas constant divided by the mixture molar mass,

.. math::
 M_\mathrm{mix} = \left(\sum_{i=1}^{n}\frac{Y_i}{M_i}\right)^{-1},
 :label: mixture_molar_mass

where :math:`M_i` is the molar mass of species :math:`i` in a mixture with :math:`n` distinct species.
Additionally, the mass fractions, of which only :math:`n-1` are independent (and only :math:`n-1` are solved for in Spitfire), are related by

.. math::
 Y_n = 1 - \sum_{i=1}^{n-1}Y_i.
 :label: Y_n_eqn


Isochoric Reactors
++++++++++++++++++

Figure :numref:`figure_isochoric_reactor_diagram` diagrams an open, constant-volume reactor with diathermal walls.
The reactor has volume :math:`V` and surface area :math:`A`.
Convective heat transfer is described by a fluid temperature :math:`T_\infty` and convective heat transfer coefficient :math:`h`.
Radiative heat transfer is determined by the temperature of the surface, :math:`T_\mathrm{surf}`, and effective emissivity, :math:`\varepsilon`.
Finally, for an isochoric reactor, mass transfer is specified by the residence time :math:`\tau`, based on volumetric flow rate, and inflowing state
with temperature :math:`T_\mathrm{in}`, density :math:`\rho_\mathrm{in}`, and mass fractions :math:`\{Y_{i,\mathrm{in}}\}`.

.. _figure_isochoric_reactor_diagram:
.. figure:: images/isochoric-reactor-diagram.png
    :width: 660px
    :align: center
    :figclass: align-center

    Isochoric reactor with mass transfer and convective and radiative heat transfer

Isochoric reactors are governed by the following equations for the reactor density, temperature, and first :math:`n-1` mass fractions.
:math:`\omega_i` is the net mass production rate of species :math:`i` due to chemical reactions,
:math:`c_v` is the specific, isochoric heat capacity of the mixture,
and :math:`e_i` and :math:`e_{i,\mathrm{in}}` are the specific internal energy of species :math:`i` in the feed and reactor.
:math:`\sigma` is the Stefan-Boltzmann constant.
We solve these equations in Spitfire to maximize sparsity and minimize calculation cost of Jacobian matrices.
Recent work [MJ2018]_ has shown that the conservation error that results from solving a temperature equation instead of an energy equation is negligible when high-order time integration methods such as those in Spitfire are used.
Closed reactors are obtained by setting :math:`\tau\to\infty`.
Adiabatic reactors are obtained by setting :math:`h,\varepsilon\to0`.
Isothermal reactors are obtained by setting the entire right-hand side of the temperature equation to zero.

.. math::
 \frac{\partial \rho}{\partial t} = \frac{\rho_\mathrm{in} - \rho}{\tau},
 :label: isochoric_rho_eqn

.. math::
 \frac{\partial Y_i}{\partial t} = \frac{\rho_\mathrm{in}}{\rho}\frac{Y_{i,\mathrm{in}} - Y_i}{\tau} + \frac{\omega_i}{\rho}, \quad i=1,\ldots,n-1
 :label: isochoric_Yi_eqn

.. math::
 \frac{\partial T}{\partial t} = \frac{\rho_\mathrm{in}}{\rho \tau c_v}\sum_{i=1}^{n}Y_{i,\mathrm{in}}(e_{i,\mathrm{in}} - e_i) - \frac{1}{\rho c_v}\sum_{i=1}^{n}\omega_i e_i + \frac{1}{\rho c_v}\frac{A}{V}\left(h(T_\infty - T) + \varepsilon\sigma(T_\mathrm{surf}^4 - T^4)\right),
 :label: isochoric_T_eqn

.. [MJ2018] Michael A. Hansen, James C. Sutherland,
    On the consistency of state vectors and Jacobian matrices,
    Combustion and Flame,
    Volume 193,
    2018,
    Pages 257-271


Isobaric Reactors
+++++++++++++++++

Figure :numref:`figure_isobaric_reactor_diagram` diagrams an open, constant-pressure reactor with diathermal walls.
The pressure, :math:`p`, of this reactor is held constant by the motion of a weightless, frictionless piston.
The expansion work done by this process is an important difference between isobaric and isochoric reactors.
We solve the following equations governing isobaric reactors.
:math:`c_p` is the specific, isobaric heat capacity of the mixture,
and :math:`h_i` and :math:`h_{i,\mathrm{in}}` are the specific internal enthalpy of species :math:`i` in the feed and reactor.

.. math::
 \frac{\partial Y_i}{\partial t} = \frac{Y_{i,\mathrm{in}} - Y_i}{\tau} + \frac{\omega_i}{\rho}, \quad i=1,\ldots,n-1
 :label: isobaric_Yi_eqn

.. math::
 \frac{\partial T}{\partial t} = \frac{1}{\tau c_p}\sum_{i=1}^{n}Y_{i,\mathrm{in}}(h_{i,\mathrm{in}} - h_i) - \frac{1}{\rho c_p}\sum_{i=1}^{n}\omega_i h_i + \frac{1}{\rho c_p}\frac{A}{V}\left(h(T_\infty - T) + \varepsilon\sigma(T_\mathrm{surf}^4 - T^4)\right),
 :label: isobaric_T_eqn

.. _figure_isobaric_reactor_diagram:
.. figure:: images/isobaric-reactor-diagram.png
    :width: 660px
    :align: center
    :figclass: align-center

    Isobaric reactor with expansion work, mass transfer, and convective and radiative heat transfer


Chemical Kinetic Models
+++++++++++++++++++++++
Spitfire currently supports various forms of reaction rate expressions for homogeneous gas-phase systems.
Let :math:`n_r` be the number of elementary reactions.
The net mass production rate of species :math:`i` is then

.. math::
    \omega_i = M_i \sum_{j=1}^{n_r}\nu_{i,j}q_j,

where :math:`\nu_{i,j}` is the net molar stoichiometric coefficient of species :math:`i` in reaction :math:`j` and :math:`q_j` is the rate of progress of reaction :math:`j`.

The rate of progress is decomposed into two parts: first, the mass action component :math:`\mathcal{R}_j`, and second, the TBAF component :math:`\mathcal{C}_j` which contains third-body enhancement and falloff effects.

.. math::
    q_j = \overset{\text{mass action}}{\mathcal{R}_j}\cdot\overset{\text{3-body + falloff}}{\mathcal{C}_j}.

The mass action component consists of forward and reverse rate constants :math:`k_{f,j}` and :math:`k_{r,j}` along with products of species concentrations :math:`\left\langle c_k\right\rangle`,

.. math::
    \mathcal{R}_j = k_{f,j}\prod_{k=1}^{N}\left\langle c_k\right\rangle^{\nu^f_{k,j}} - k_{r,j}\prod_{k=1}^{N}\left\langle c_k\right\rangle^{\nu^r_{k,j}},

in which :math:`\nu^f_{i,j}` and :math:`\nu^r_{i,j}` are the forward and reverse stoichiometric coefficients of species :math:`i` in reaction :math:`j`, respectively.

The forward rate constant is found with a modified Arrhenius expression,

.. math::
    k_{f,j} = A_j T^{b_j} \exp\left(-\frac{E_{a,j}}{R_u T}\right) = A_j T^{b_j} \exp\left(-\frac{T_{a,j}}{T}\right),

where :math:`A_j`, :math:`b_j`, and :math:`E_{a,j}` are the pre-exponential factor, temperature exponent, and activation energy of reaction :math:`j`, respectively.
We define :math:`T_{a,j}=E_{a,j}/R_u` as the activation temperature.

The reverse rate constant of an irreversible reaction is zero.
:math:`k_{r,j}` for a reversible reaction is found with the equilibrium constant :math:`K_{c,j}`, via :math:`k_{r,j} = k_{f,j}/K_{c,j}`.
The equilibrium constant is

.. math::
    K_{c,j} = \left(\frac{p_\text{atm}}{R_u}\right)^{\Xi_j}\exp\left(\sum_{k=1}^{N}\nu_{k,j}B_k\right),

where :math:`\Xi_j=\sum_{k=1}^{N}\nu_{k,j}` and :math:`B_k` is

.. math::
    B_k = -\ln(T) + \frac{M_k}{R_u}\left(s_k - \frac{h_k}{T}\right).

For the TBAF component :math:`\mathcal{C}_j` there are two nontrivial cases: (1) a three-body reaction and, (2) a unimolecular/recombination falloff reaction.
If a reaction is not of a three-body or falloff type, then :math:`\mathcal{C}_j = 1`.
For three-body reactions, it is

.. math::
    \mathcal{C}_j = \left\langle c_{TB,j}\right\rangle = \sum_{i=1}^{N}\alpha_{i,j}\left\langle c_i\right\rangle,

where :math:`\alpha_{i,j}` is the third-body enhancement factor of species :math:`i` in reaction :math:`j`, and :math:`\left\langle c_{TB,j}\right\rangle` is the third-body-enhanced concentration of reaction :math:`j`.
The quantity :math:`\alpha_{i,j}` defaults to one if not specified.
For falloff reactions, the TBAF component is

.. math::
    \mathcal{C}_j = \frac{p_{fr,j}}{1 + p_{fr,j}}\mathcal{F}_j,

in which :math:`p_{fr,j}` and :math:`\mathcal{F}_j` are the falloff reduced pressure and falloff blending factor, respectively.
The falloff reduced pressure is

.. math::
    p_{fr,j} = \frac{k_{0,j}}{k_{f,j}}\mathcal{T}_{F,j},

where :math:`k_{0,j}` is the low-pressure limit rate constant evaluated with low-pressure Arrhenius parameters :math:`A_{0,j}`, :math:`b_{0,j}`, :math:`E_{a,0,j}`, and :math:`\mathcal{T}_{F,j}` is the concentration of the mixture
which is either that of a single species if specified or the third-body-enhanced concentration if not.

The falloff blending factor :math:`\mathcal{F}_j` depends upon the specified falloff form.
For the Lindemann approach, :math:`\mathcal{F}_j = 1`.
In the Troe form,

.. math::
    \mathcal{F}_j &= \mathcal{F}_{\text{cent}}^{1/(1+(A/B)^2)}, \\
    \mathcal{F}_{\text{cent}} &= (1-a_{\text{Troe}})\exp\left(-\frac{T}{T^{***}}\right) + a_{\text{Troe}}\exp\left(-\frac{T}{T^{*}}\right) + \exp\left(-\frac{T^{**}}{T}\right), \\
    A &= \log_{10}p_{FR,j} - 0.67\log_{10}\mathcal{F}_{\text{cent}} - 0.4, \\
    B &= 0.806 - 1.1762\log_{10}\mathcal{F}_{\text{cent}} - 0.14\log_{10}p_{FR,j},


where :math:`a_{\text{Troe}}`, :math:`T^{*}`, :math:`T^{**}`, and :math:`T^{***}` are specified parameters of the Troe form.
If :math:`T^{***}` is unspecified in the mechanism file then its term is ignored.


todo: add description of new non-elementary reaction rates


Species Thermodynamics
++++++++++++++++++++++
Spitfire supports ideal mixtures of thermally perfect species, and species heat capacity may be modeled as a constant or through a NASA-7 or NASA-9 polynomial.
NASA-7 polynomials must be used with two temperature regions, while any number of regions may be used with a NASA-9 polynomial.










