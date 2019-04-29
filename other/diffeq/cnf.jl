using OrdinaryDiffEq
using Distributions
using Flux, DiffEqFlux, ForwardDiff, Tracker

# Neural Network
nn = Chain(Dense(1,1,tanh))
p = DiffEqFlux.destructure(nn)
tspan = Float32.((0.0, 10.0))
function cnf(u,p,t)
  z = @view u[1:end-1,:]
  m = DiffEqFlux.restructure(nn,p)
  jac = [-sum(Tracker.jacobian((zi)->log.(zi),[z[i]])) for i in eachindex(z)]
  if u isa TrackedArray
      res = Tracker.collect(cat(m(z),jac,dims=1))
  else
      res = Tracker.data.(cat(m(z),jac,dims=1))
  end
  res
end

prob = ODEProblem(cnf,nothing,tspan,nothing)
params = Params([p])

function predict_adjoint(x)
    diffeq_adjoint(p,prob,Tsit5(),u0=[x;zero(x)],
                   saveat=0.0:0.1:10.0,
                   sensealg=DiffEqFlux.SensitivityAlg(quad=false,
                                backsolve=true,autojacvec=true))
end

function loss_adjoint(xs)
    pz = Normal(0.0, 1.0)
    preds = predict_adjoint(xs)[:,:,end]
    z = preds[1:end-1,:]
    delta_logp = preds[end:end,:]

    logpz = logpdf.(pz, z)
    logpx = logpz - delta_logp
    loss = -mean(logpx)
end

opt = ADAM(0.1)

const BATCH_SIZE = 100
raw_data = Float32.(rand(Normal(2.0, 0.1),BATCH_SIZE))' # (D,BS)
data = Iterators.repeated([raw_data], 1);

loss_adjoint(raw_data)

Flux.train!(loss_adjoint, params, data, opt)
iszero(Tracker.grad(nn[1].W))
