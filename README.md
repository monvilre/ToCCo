# ToCCo
Linear model of a flow over a sinusoidal topography
ToCCo,  for "Topographic Coupling at Core-Mantle interface" is distributed under the CeCILL License. ToCCo is a local model that aims to calculate flows over a topography or confined between two boundaries. It uses a linear perturbative approach to solve Magneto-Hydro-Dynamic equations. It can, therefore, take into consideration: rotation, magnetic field, stratification, and fluid viscosity.
The most common boundary conditions can be selected. Are implemented, inviscid no-penetration, viscous no-slip, and stress-free conditions. It is also possible to oscillate the boundaries, inserting time dependence.
The final step of the code is to compute the mean shear stress on the boundary from pressure and magnetic fields.\\

ToCCo is coded in Python language, it is a hybrid code that combines symbolic mathematics with the Sympy library \href{https://www.sympy.org/}{https://www.sympy.org/} and numerical evaluation with mpmath library \href{http://mpmath.org/}{http://mpmath.org/}, which provides arbitrary-precision floating-point calculation.

It is provided with visualization tools which allows to visualize 2D cross-section with Matplotlib (\href{https://matplotlib.org/}{https://matplotlib.org/}), or full 3D field with Paraview (\href{https://www.paraview.org/}{https://www.paraview.org/}).\\
