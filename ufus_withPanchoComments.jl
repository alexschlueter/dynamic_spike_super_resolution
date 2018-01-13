push!(LOAD_PATH, "./models")
using SuperResModels
using SparseInverseProblems

## For publishing, eliminate all comments with doble hashtag. These are for those that are not familiar with Julia.

## The everywhere method obligates all process to load this code, this is for parallel computing.
@everywhere begin
    include("ufus_parameters.jl")
    parameters = SuperResModels.Conv2dParameters(x_max, x_max, filter, filter_dx, filter_dy, sigma, sigma, sigma, sigma)
    model_static = SuperResModels.Conv2d(parameters)
    dynamic_parameters = SuperResModels.DynamicConv2dParameters(K, tau, v_max, 20)
    model_dynamic = SuperResModels.DynamicConv2d(model_static, dynamic_parameters)
    n_x = model_static.n_x
    n_y = model_static.n_y
end

### Get a string with the time of when the script started, to name experiments ###
now_str = string(now())
now_str = replace(now_str, ":", "-")
now_str = replace(now_str, ".", "-")

### L2 norm single particle (probably to just use this value) ###
# thetas stands for the location of the particles in Omega
# In this case, we are static, so the particles are just 
# 2d vectors.
thetas = reshape([x_max/2; x_max/2], 2, 1)
# With its corresponding weights.
weights = [1.0]
# phi requires to know which type of particles it is, and its information: location and weight.
# Psi represents the forward operator for a single particle
# Phi represents the weighted sum of these psi with their weights.
single_particle_norm = norm(phi(model_static, thetas, weights), lp_norm)
@everywhere single_particle_norm = $single_particle_norm
println("norm = ", single_particle_norm)

#######################################
# As a reminder: The examples we seek to emulate correspond to two sets of particles moving along a two stripes on each side.
# So particles_m stands for those located on the west side, moving towards the south. 
# The vector particles_m basically are the locations at time 0 of the particles, it also defines quantity.
# Particles_p are the symmetric equivalent, on the east and moving towards the north. 
#######################################

### Test dynamic ###
println("Testing dynamic...")	
particles_m = [x_max/4]
# To test the other group of particles: particles_p = [3*x_max/4]
thetas = hcat([(x_max/2 - dx) * ones(1, length(particles_m)); particles_m'; 0.0; -v_max/2])
#[(x_max/2 + dx) * ones(1, length(particles_p)); particles_p'; 0.0; v_max/2])
weights = ones(1)
target = phi(model_dynamic, thetas, weights)
# Function required by the method SparseInverseProblems.ADCG to solve our minimization problem.
function callback(old_thetas, thetas, weights, output, old_obj_val)
    #evalute current OV
    new_obj_val,t = SparseInverseProblems.loss(SparseInverseProblems.LSLoss(), output - target)
    #println("gap = $(old_obj_val - new_obj_val)")
    if old_obj_val - new_obj_val < 1E-4
       	return true
    end
    return false
end
# The 4th input in this method is the bound on the total variation of the measure. This is basically a bound on the sum of the weights.
(thetas_est,weights_est) = SparseInverseProblems.ADCG(model_dynamic, SparseInverseProblems.LSLoss(), target, 2.0, callback=callback, max_iters=200)
println("theta = ", thetas, ", weights = ", weights)
println("theta_est = ", thetas_est, ", weights_est = ", weights_est)

### Generate sequence ###
println("Generating sequence...")
## SharedArray makes a global variable that can be accessed by all the processors.
## Global variable describing the dynamic particles by static screenshots.
video = SharedArray{Float64}(n_x * n_y, n_im)
particles_m = [x_max/4]
particles_p = [3*x_max/4]

# p is the probability for new particles to activate or deactivate.
p=0.02
for i in 1:n_im
    # Print current time sample
    println(i, "/", n_im) 
    # Obtain the current static particles, just spatial, there is no speed.
    thetas = hcat([(x_max/2 - dx) * ones(1, length(particles_m)); particles_m'],
                  [(x_max/2 + dx) * ones(1, length(particles_p)); particles_p'])
    weights = ones(size(thetas, 2))
    # Image it and add it to our data vector: video
    if (length(weights) > 0)
        video[:,i] = phi(model_static, thetas, weights)
        video[:,i] = video[:,i] + sigma_noise * randn(size(video[:,i]))
    end
    # Displace the particles by their speeds
    particles_m -= v_max/2 * tau
    particles_p += v_max/2 * tau
    # Process in which particles are deactivated
        remove = []
        for i in 1:length(particles_m)
            if rand() < p
                 push!(remove, i)
            end
        end
        deleteat!(particles_m, remove)
        remove = []
        for i in 1:length(particles_p)
        if rand() < p
            push!(remove, i)
        end
	end
	deleteat!(particles_p, remove)
    # Process in which particles are activated
	if length(weights) == 0
	    push!(particles_m, initial_position_generator())
	    push!(particles_p, initial_position_generator())
        end
        if rand() < p
            push!(particles_m, initial_position_generator())
        end
        if rand() < p
            push!(particles_p, initial_position_generator())
        end
end


### Obtaining time frames in which the quantity of particles remained constant ###
println("Getting short sequences without jump...")
# get the total mass at each time step
frame_norms = [norm(video[:, i], lp_norm) for i in 1:n_im]
@everywhere frame_norms = $frame_norms
println("norms: ", frame_norms)
# Find locations in which there was a significative mass difference between two time steps.
jumps = find(abs.(frame_norms[2:end] - frame_norms[1:end-1]) .> jump_threshold*single_particle_norm)
jumps = [0; jumps; n_im]
# Obtain non-overlapping intervals of 2K+1 consecutive time samples in which the mass didn't changed.
short_seqs = []
for i in 1:length(jumps)-1
    append!(short_seqs, [(jumps[i] + 5*j + 1):(jumps[i] + 5*j + 5) for j in 0:(div(jumps[i+1] - jumps[i], 5) -1)])
end
println("jumps: ", jumps)
println("short seqs: ", short_seqs)

### Function that given the a subquence of the total video, will estimate the locations and weights of the
### involved particles. 

@everywhere function posvel_from_seq(video, seq)
    assert(length(seq) == 5)
    target = video[:,seq][:]
    # estimated number of particles in the sequence.
    est_num_particles = div(frame_norms[seq[1]], single_particle_norm*0.95)
    if (est_num_particles == 0)
        return Matrix{Float64}(5,0)
    end
    # Function required to use the SparseInverseProblems.ADCG method.
    function callback(old_thetas, thetas, weights, output, old_obj_val)
        #evalute current OV
        new_obj_val,t = SparseInverseProblems.loss(SparseInverseProblems.LSLoss(), output - target)
        #println("gap = $(old_obj_val - new_obj_val)")
        if old_obj_val - new_obj_val < 1E-4
            return true
        end
        return false
    end
    # It uses the ACDG algorithm to estimate the location and weights of the particles. Using the estimated number of particles we can bound the total variation on the solutions.
    (thetas_est,weights_est) = SparseInverseProblems.ADCG(model_dynamic, SparseInverseProblems.LSLoss(), target, frame_norms[seq[1]], callback=callback, max_iters=2000)
    if length(thetas_est) > 0
        println("est_num = ", est_num_particles)
        println("thetas = ", thetas_est) 
        println("weights = ", weights_est)
        return [thetas_est; weights_est']
    else
        return Matrix{Float64}(5,0)
    end
end

### We solve our inverse problem in a parallel fashion, for all the found sequences ###
println("Inverting...")
## pmap is parallel computing map, the map function receives as input a function and a vector, it applies to function to each element of the vector
## pmap does exactly the same, but in a parallel fashion. all_thetas is an array that at each position, has a [thetas_est; weights_est] matrix.
all_thetas = pmap(seq -> posvel_from_seq(video, seq), short_seqs)

println("Reprojecting...")
errors = zeros(length(short_seqs))
### Measurements error, we simulate the measurements that would be obtained with our reconstructed values ###
for i in 1:length(short_seqs)
    seq=  short_seqs[i]
    ## Extracts the submatrix from the video, and converts it into a vector.
    target = video[:, seq][:]
    if length(all_thetas[i]) > 0
	## all_thetas[i][1:4,:] gives the positions in space x space x velocity x velocity
	## The remaining row describes the weights of the reconstructions.
        reprojection = phi(model_dynamic, all_thetas[i][1:4,:], all_thetas[i][5,:])
        println("error = ", norm(target-reprojection))
        errors[i] = norm(target-reprojection)
    else
        errors[i] = norm(target)
    end
end

### Save the simulated data ###
using PyCall
@pyimport numpy as np
mkdir(now_str)
cp("ufus_parameters.jl", string(now_str, "/ufus_parameters.jl"))
cd(now_str)
short_seq_array = hcat(short_seqs...)
np.save("video", video)
np.save("frame_norms", frame_norms)
np.save("jumps", jumps)
np.save("short_seq_array", short_seq_array)
for i in 1:length(short_seqs)
    np.save(string("thetas-", i), all_thetas[i])
end
np.save("errors", errors)
cd("..")