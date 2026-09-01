# Supercell convergence

`melting.supercell_convergence` runs direct `melting.calphy` calculations over
isotropic supercells. It accepts only `calphy` or `melting.calphy` as
`inner_method`; its `inner_method_parameters` are forwarded unchanged except
that the wrapper owns `supercell`. Supplying an inner `supercell` is rejected.

The default initial sizes are 6 and 7, meaning `[6, 6, 6]` and `[7, 7, 7]`.
With `parallel_initial=True` (the default), those children submit together;
set it false where resources require serial execution. Later sizes are evaluated
one at a time through `maximum_size` (default 9).

Consecutive valid child temperatures are compared using
`100 * 2 * abs(T_b - T_a) / (T_b + T_a)`. A pair within
`relative_tolerance_percent` selects the larger child and preserves its
`success` or `ambiguous` status. Scientific child `unconverged` results are
recorded but excluded from comparisons. If no pair meets tolerance, the largest
valid result (or largest provisional result) is returned as `unconverged`.

The report retains selected Calphy metadata and includes a `convergence` payload
with each tested child, exclusions, and pairwise comparisons. This finite-size
check is not a claim of experimental accuracy.
