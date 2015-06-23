module SchattenNorms

export snorm, nucnorm, trnorm, specnorm

function nucnorm{T}(m::AbstractMatrix{T})
    snorm(m,1)
end

trnorm(m) = nucnorm(m)

function specnorm{T}(m::AbstractMatrix{T})
    _,s,_ = svd(m)
    return maximum(s)
end

function snorm{T}(m::AbstractMatrix{T},p=2.0)
    p == 2.0 && return vecnorm(m,2)
    p == Inf && return specnorm(m)
    _,s,_ = svd(m)
    return norm(s,p)
end


end # module
