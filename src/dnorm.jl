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

using Convex

"""
 ϕ represents the isomorphism between complex and real matrices.
 e.g., see [Invariant semidefinite programs](http://arxiv.org/abs/1007.2905)
 by Bachoc et al.  If two arguments are given, they are taken to be the real and
 imaginary parts of a complex matrix.
"""
function ϕ(r,i)
    if size(r) != size(i)
        error("ϕ requires both arguments to be of the same size.")
    end
    [r i; -i r]
end

function ϕ(c)
    r = real(c)
    i = imag(c)
    ϕ(r,i)
end

"""
Extract the real part of a complex matrix represented as a real matrix
"""
function ϕr(m)
    m[1:div(end,2),1:div(end,2)]
end

"""
Extract the imag part of a complex matrix represented as a real matrix.
"""
function ϕi(m)
    m[1:div(end,2),div(end,2)+1:end]
end

"""
Extract the real and imag parts of a complex matrix represented as a real matrix.
"""
function ϕinv(m)
   ϕr(m) + 1im*ϕi(m)
end

"""
Compute the trace of the real representation of a complex matrix.
"""
function retrϕ(m)
    trace(ϕr(m))
end

function ket(i,d)
    v = spzeros(Float64,d,1)
    v[i+1] = 1.0
    return v
end

function bra(i,d)
    return ket(i,d)'
end

"""
Generates linear map E_ such that E_(ρ) → 1 ⊗ ρ
"""
function E_(id_dim, ρ_dim)
    M = spzeros(Float64,id_dim^2*ρ_dim^2,ρ_dim^2)
    for m in 0:ρ_dim-1
        for n in 0:ρ_dim-1
            for k in 0:id_dim-1
                M += kron(ket(k,id_dim),ket(m,ρ_dim),ket(k,id_dim),ket(n,ρ_dim))*kron(bra(m,ρ_dim),bra(n,ρ_dim))
            end
        end
    end
    return M
end

"""
Permutes the elements of a matrix so that it transforms the column
major representation of a linear map L into a matrix C that is
positive iff L is completely positive, Hermitian iff L maps
(vectorized) Hermitian matrices to Hermitian matrices. In other words,
it corresponds to the Choi-Jamiolkoski isomorphism.
"""
function involution(m)
    dsq = size(m,1) # we assume the matrix is square
    d   = Int(sqrt(dsq))
    return reshape(permutedims(reshape(m,(d,d,d,d)),[2,4,1,3]),(dsq,dsq))
end

let # wat13b
    global dnorm
    local prev_dx, M

    prev_dx = -1

    """
    dnorm(L) 

    Computes the diamond norm of a linear superoperator `L` (i.e., a
    linear transformation of operators). The superoperator must be
    represented in column major form. In other words, it must be given
    by a matrix that, when multiplying a vectorized (column major)
    operator, it should result in the vectorized (column major)
    representation of the result of the transformation.
    """
    function dnorm(L)
        J = involution(L)

        dx = size(J,1) |> sqrt |> x -> round(Int,x)
        dy = dx

        if prev_dx != dx
            M = E_(dy,dx)
            prev_dx = dx
        end

        Jr = real(J)
        Ji = imag(J)

        Xr  = Variable(dy*dx, dy*dx)
        Xi  = Variable(dy*dx, dy*dx)
        ρ0r = Variable(dx, dx)
        ρ0i = Variable(dx, dx)
        ρ1r = Variable(dx, dx)
        ρ1i = Variable(dx, dx)

        prob = maximize( trace( Jr * Xr + Ji * Xi ) )

        prob.constraints += trace(ρ0r) == 1
        prob.constraints += trace(ρ0i) == 0
        prob.constraints += trace(ρ1r) == 1
        prob.constraints += trace(ρ1i) == 0

        Mρ0r = reshape(M * vec(ρ0r), dy*dx, dy*dx)
        Mρ0i = reshape(M * vec(ρ0i), dy*dx, dy*dx)
        Mρ1r = reshape(M * vec(ρ1r), dy*dx, dy*dx)
        Mρ1i = reshape(M * vec(ρ1i), dy*dx, dy*dx)

        prob.constraints += isposdef( ϕ(ρ0r,ρ0i) )

        prob.constraints += isposdef( ϕ(ρ1r,ρ1i) )

        prob.constraints += isposdef( ϕ( [ Mρ0r Xr ; Xr' Mρ1r ], [ Mρ0i Xi ; -Xi' Mρ1i ] ) )

        solve!(prob)

        if prob.status != :Optimal
            #println("DNORM warning.")
            #println("Input: $(L)")
            #println("Input's Choi spectrum: $(eigvals(liou2choi(L)))")
            warn("Diamond norm calculation did not converge.")
        end

        return prob.optval
    end
end
