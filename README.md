# relax_disloc
## Simulating `Relax` displacements with dislocations

These notebooks use postseismic displacement fields calculated using `Relax`[^1][^2] to constrain a distribution of slip on a network of dislocations representing the crust-mantle interface (CMI).

The main notebooks are: 
- [relax_disloc_param_sweep_final.ipynb](relax_disloc_param_sweep_final.ipynb): Estimates CMI slip from displacement fields arising from combinations of source fault dip, thickness of elastic layer, and viscosity
- [relax_disloc_inter_final.ipynb](relax_disloc_inter_final.ipynb): Estimates slip on CMI and source fault elements from a composite displacement field representing postseismic and interseismic deformation
- [relax_disloc_chichi.ipynb](relax_disloc_chichi.ipynb): Simulates postseismic deformation models of [_Rousset et al._ (2012)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2012JB009571)

The first two notebooks require result files generated from a suite of `Relax` models. Once `Relax` is installed and ready to run, change to the `parameter_ranges` directory and run `sh run_all.sh`

[^1]: [GitHub](https://github.com/geodynamics/relax/tree/master)
[^2]: [binaries@CIG](https://geodynamics.org/resources/relax/supportingdocs)
