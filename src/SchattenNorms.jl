#    Copyright 2015 Raytheon BBN Technologies
#  
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
#  
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

module SchattenNorms

export snorm, nucnorm, trnorm, specnorm, fnorm, dnorm

"""
Computes the nuclear norm of a matrix `m`.
"""
function nucnorm{T}(m::AbstractMatrix{T})
    snorm(m,1)
end

"""
Computes the trace norm of a matrix `m`.
"""
trnorm(m) = nucnorm(m)

"""
Computes the Frobenius norm of a matrix `m`.
"""
fnorm(m) = vecnorm(m,2)

"""
Computes the spectral norm of a matrix `m` (i.e., the maximum singular value).
"""
function specnorm{T}(m::AbstractMatrix{T})
    _,s,_ = svd(m)
    return maximum(s)
end

"""
Computes the `p`-Schatten norm of a matrix `m`.
"""
function snorm{T}(m::AbstractMatrix{T},p=2.0)
    p == 2.0 && return vecnorm(m,2)
    p == Inf && return specnorm(m)
    _,s,_ = svd(m)
    return norm(s,p)
end

include("dnorm.jl")

end # module
