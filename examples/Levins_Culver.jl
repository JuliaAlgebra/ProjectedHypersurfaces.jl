using Random, Plots, LinearAlgebra, ProjectedHypersurfaces

mkpath("./results/quadratic");

Random.seed!(1234)

# Incidence variety of the discriminant
@var y z α β γ δ
steady_state = [β*y*(1-y)-γ*y;
δ*z*(1-y-z)-β*y*z-α*z]
Jac = differentiate(steady_state, [y,z])
detJac = det(Jac)
F = System([steady_state; detJac], variables = [α; β; γ; δ; y;z])


# Set up projected hypersurface
h = ProjectedHypersurface(F, [α, β, γ, δ])

# Degree of the discriminant
d = degree(h)
println("Degree of discriminant: $d")

# Pick a center for the routing function
c = [13, 2, 0.5, 7]

# Set up the routing function gradient
r = RoutingFunction(h; c=c)

# Find the complex critical points 
# pts = read_solutions("./results/quadratic/routing_points.txt") |> real
routing_result = critical_points(r)
pts = routing_points(routing_result)
res = result(routing_result)
mon_res = monodromy_result(routing_result)

write_parameters("./results/levins_culver/monodromy_parameters.txt", parameters(mon_res))
write_solutions("./results/levins_culver/monodromy_result.txt", solutions(mon_res))
write_solutions("./results/levins_culver/result.txt", solutions(res))
write_solutions("./results/levins_culver/routing_points.txt", pts)

# Connect the critical points
roadmap = gradient_roadmap(r, routing_result)
G = partition(roadmap)
idx = morse_indices(roadmap)
failures = failed_info(roadmap)
println("Connected components: $(G)")
println("Indicies: $(idx)")
println("Failed info: $(failures)")
println()

write("./results/levins_culver/connected_components.txt", string(G))

