using Random, Plots, LinearAlgebra,  ImplicitPlots, ProjectedHypersurfaces

Random.seed!(123)

# Set up incidence variety of discriminant
@var r d a g A M C γ 
steady_state = [a*M*C-g*M*A+γ*M*(1-M-C);
    r*C*(1-M-C)-d*C-a*M*C;
    A*(1-C)-1
]
Jac = differentiate(steady_state, [M, C, T])
detJac = det(Jac)
F = System([steady_state; detJac], variables = [r; d; a; g; γ; A; M; C])

# Set up the projected hypersurface
h_original = ProjectedHypersurface(F, [b; c; d; f; βy; βz; γy;γz;])

# Degree of the original discriminant
d_original = degree(h_original)

# Fix a center for the routing functions
C = [0.02, 0.2, 0.05,0.5,0.4,0.04,0.98,0.07]
r = RoutingFunction(h_original; c=C)

routing_result = critical_points(r; start_grid_width=0)
pts = routing_points(routing_result)
res = result(routing_result)
mon_res = monodromy_result(routing_result)

# Connect the critical points
roadmap = gradient_roadmap(r, routing_result)
G = partition(roadmap)
idx = morse_indices(roadmap)
failures = failed_info(roadmap)

#pl_original_smaller = generate_plot(
#    r_original, pts_original,
#    G_original,
#    idx_original;
#    contour_stepsize = 5e-4, contour_levels = 60, markersize = 6, legend = false,
#    xlims = (0, 0.09), ylims = (0, 0.5),
#    root_count_fn = allee_root_count_fn,
#    annotate_root_counts = true, annotation_textsize = 4,
#)
#savefig("figures/tripartite.png")
#savefig("figures/tripartite.pdf")
#savefig("figures/tripartite.svg")

