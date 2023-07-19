export OptimizationProblem, primalDualSolve
######
# Simple Primal-Dual IP for the problem
#
# min_x f(x)
#   Cx -h <= 0.0

struct OptimizationProblem
  d :: Int64
  m :: Int64
  f_g_h :: Function
  #R^d |----> R, returns a tuple with function value, gradient, hessian
  C :: Matrix{Float64} #C \in R^(d,m)
  h :: Vector{Float64}
end

# BV page 610
# The newton system is
# [\nabla^2 f_0, C' ]
# [-diag(\lambda) *C, -diag(C*x -h ) ]

function primalDualSolve(p :: OptimizationProblem, x :: Vector{Float64}, lambda :: Vector{Float64}, mu :: Float64, e_feas :: Float64, e :: Float64, max_iters :: Int64, alpha :: Float64, beta :: Float64)
  @assert(mu > 1)
  @assert(0.0 < beta < 1.0)
  @assert(p.d == length(x))
  @assert(p.m == length(lambda))
  @assert(all(p.C*x .<= p.h))
  @assert(all(lambda .> 0.0))
  @assert(e > 0.0 && e_feas > 0.0)
  eta_hat(x,lambda) = -dot(lambda,p.C*x-p.h)

  function compute_r_t(f,g,x,lambda,t)
    [g + p.C'*lambda;
    -lambda.*(p.C*x - p.h) - (1.0/t)*ones(p.m)]
  end

  iter = 1
  for iter = 1:max_iters
    # println("ITER :", iter, ". Printing values of x: ", x)
    e_hat = eta_hat(x,lambda)
    t = (p.m*mu)/e_hat
    (f,g,h) = p.f_g_h(x)
    @assert(length(g) == p.d)
    @assert(size(h) == (p.d,p.d))

    #check termination conditions... Here?
    r_old = norm(compute_r_t(f,g,x,lambda,t))
    if (iter == max_iters)
      println(norm(g + p.C'*lambda))
      println(e_hat)
      println()
    end
    if (norm(g + p.C'*lambda) < e_feas && e_hat < e)
      break;
    end
    #form newton system...
    newton_system = [h p.C';
      -diagm(lambda)*p.C -diagm(p.C*x - p.h)]
    #target
    r_dual = g + p.C'*lambda
    r_cent = -lambda.*(p.C*x - p.h) - (1.0/t)*ones(p.m)
    #solve
    search_direction = -(newton_system\[r_dual; r_cent])
    sd_x = search_direction[1:p.d]
    # println(" Printing values of sd_x:  ", sd_x)
    sd_lambda = search_direction[p.d+1:end]
    #linesearch
    neg_inds = findall(sd_lambda .< 0.0) #bullshit
    s_max = 1.0
    if length(neg_inds) > 0
      s_max = min(1.0, minimum(-lambda[neg_inds]./sd_lambda[neg_inds]))
    end
    #hardcoded in BV
    s = 0.995*s_max
    #backtracking for feasibility
    loopingIndex2 = 1
    while(!all(p.C*(x + s*sd_x) .<= p.h ))
      s = beta*s
      if loopingIndex2 ==1000
	  println("########################################################")
	  println("########################################################")
	  println("### WARINING ENTERING INFINITE LOOP, EXIT WITH ERROR ###")
    println("########################################################")
	  println("########################################################")
	  println("########################################################")
	  error()
      end
      loopingIndex2 = loopingIndex2 + 1
    end

    #backtracking for some other thing
    loopingIndex = 1
    while(true)
      if mod(loopingIndex,100) == 0
        println("### loopin' ",loopingIndex," !########")
      end
      loopingIndex = loopingIndex + 1
      x_plus = x + s*sd_x
      lambda_plus = lambda + s*sd_lambda
      f_t,g_t,h_t = p.f_g_h(x_plus) #only need f_g
      #compute norm of r_t
      r_t = compute_r_t(f_t,g_t,x_plus,lambda_plus,t)
      if (norm(r_t) <= (1-alpha*s)*r_old)
        break;
      end
      s = beta*s
    end
    x = x + s*sd_x
    lambda = lambda + s*sd_lambda
  end
  if iter == max_iters
    @warn("Hit max iters in interior point method!")
  end
  return x,lambda
end
