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
using SCS

import SparseArrays: spzeros
import LinearAlgebra: tr, opnorm

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
    tr(ϕr(m))
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
E_(id_dim, ρ_rim)

Generates linear map E such that E(ρ) → 1 ⊗ ρ

Note that the dual of this map is F such that F*vec(σ ⊗ ρ) → trace(σ) vec(ρ),
in other words, E is the dual of the partial trace.

"""
function E_(id_dim, ρ_dim)
    M = spzeros(Float64,id_dim^2*ρ_dim^2,ρ_dim^2)
    for m in 0:ρ_dim-1
        for n in 0:ρ_dim-1
            for k in 0:id_dim-1
                M += Base.kron(ket(k,id_dim),ket(m,ρ_dim),ket(k,id_dim),ket(n,ρ_dim))*Base.kron(bra(m,ρ_dim),bra(n,ρ_dim))
            end
        end
    end
    return M
end

"""
involution(m)

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

let # wat09b
    global dnormcptp2
    local prev_dy, F

    prev_dy = -1

    """
    dnormcptp(L1,L2)

    Computes the diamond norm of a linear superoperator `L` (i.e., a
    linear transformation of operators). The superoperator must be
    represented in column major form. In other words, it must be given
    by a matrix that, when multiplying a vectorized (column major)
    operator, it should result in the vectorized (column major)
    representation of the result of the transformation.
    """
    function dnormcptp2(L1,L2)
        J = involution(L1-L2)

        dx = size(J,1) |> sqrt |> x -> round(Int,x)
        dy = dx

        if prev_dy != dy
            F = E_(dy,dx)
            prev_dy = dy
        end

        Jr = real(J)
        Ji = imag(J)

        Wr = Variable(dy*dx, dy*dx)
        Wi = Variable(dy*dx, dy*dx)

        Mr = Variable(dy*dx, dy*dx)
        Mi = Variable(dy*dx, dy*dx)

        ρr = Variable(dx, dx)
        ρi = Variable(dx, dx)

        prob = maximize( tr( Jr*Wr + Ji*Wi ) )

        prob.constraints += tr(ρr) == 1
        prob.constraints += tr(ρi) == 0

        Mr = reshape(F*vec(ρr), dy*dx, dy*dx)
        Mi = reshape(F*vec(ρi), dy*dx, dy*dx)

        prob.constraints += isposdef( ϕ(ρr,ρi) )
        prob.constraints += isposdef( ϕ(Wr,Wi) )
        prob.constraints += isposdef( ϕ(Mr,Mi) - ϕ(Wr,Wi) )

        solve!(prob, SCSSolver(verbose=0, eps=1e-6, max_iters=5_000))

        if prob.status != :Optimal
            #println("DNORM_CPTP warning.")
            #println("Input: $(L)")
            #println("Input's Choi spectrum: $(eigvals(liou2choi(L)))")
            warn("Diamond norm calculation did not converge.")
        end

        return 2*prob.optval
    end
end

let # wat09b
    global dnormcptp
    local prev_dy, F

    prev_dy = -1

    """
    dnormcptp(L1,L2)

    Computes the diamond norm distance between two linear completely
    positive and trace preserving superoperators `L1` and `L2` . The
    superoperators must be represented in column major form.

    """
    function dnormcptp(L1,L2)
        J = involution(L1-L2)

        dx = size(J,1) |> sqrt |> x -> round(Int,x)
        dy = dx

        if prev_dy != dy
            F = E_(dy,dx)'
            prev_dy = dy
        end

        Jr = real(J)
        Ji = imag(J)

        Zr = Variable(dy*dx, dy*dx)
        Zi = Variable(dy*dx, dy*dx)

        pZr = reshape(F*vec(Zr), dx, dx)
        pZi = reshape(F*vec(Zi), dx, dx)

        prob = minimize( opnorm( ϕ(pZr, pZi) ) )

        prob.constraints += isposdef( ϕ(Zr,Zi) )
        prob.constraints += isposdef( ϕ(Zr,Zi) - ϕ(Jr,Ji) )

        solve!(prob, SCSSolver(verbose=0, eps=1e-6, max_iters=5_000))

        if prob.status != :Optimal
            #println("DNORM_CPTP warning.")
            #println("Input: $(L)")
            #println("Input's Choi spectrum: $(eigvals(liou2choi(L)))")
            warn("Diamond norm calculation did not converge.")
        end

        return 2*prob.optval
    end
end

let # wat13b
    global dnorm2
    local prev_dy, F

    prev_dy = -1

    """
    dnorm(L1,L2)

    Computes the diamond norm of a linear superoperator `L` (i.e., a
    linear transformation of operators). The superoperator must be
    represented in column major form. In other words, it must be given
    by a matrix that, when multiplying a vectorized (column major)
    operator, it should result in the vectorized (column major)
    representation of the result of the transformation.
    """
    function dnorm2(L1,L2)
        J = involution(L1-L2)

        dx = size(J,1) |> sqrt |> x -> round(Int,x)
        dy = dx

        if prev_dy != dy
            F = E_(dy,dx)'
            prev_dy = dy
        end

        Jr = real(J)
        Ji = imag(J)

        Y0r = Variable(dy*dx, dy*dx)
        Y0i = Variable(dy*dx, dy*dx)

        Y1r = Variable(dy*dx, dy*dx)
        Y1i = Variable(dy*dx, dy*dx)

        prob = minimize( opnorm(ϕ( ρ0r, ρ0i )) + opnorm(ϕ( ρ1r, ρ1i )))

        ρ0r = reshape(F*vec(Y0r), dx, dx)
        ρ0i = reshape(F*vec(Y0i), dx, dx)

        ρ1r = reshape(F*vec(Y1r), dx, dx)
        ρ1i = reshape(F*vec(Y1i), dx, dx)

        prob.constraints += isposdef( ϕ(Y0r,Y0i) )
        prob.constraints += isposdef( ϕ(Y1r,Y1i) )
        prob.constraints += isposdef( ϕ( [ Y0r -Jr ; -Jr' Y1r ], [ Y0i -Xi ; Xi' Y1i ] ) )

        solve!(prob, SCSSolver(verbose=0, eps=1e-6, max_iters=5_000))

        if prob.status != :Optimal
            #println("DNORM_CPTP warning.")
            #println("Input: $(L)")
            #println("Input's Choi spectrum: $(eigvals(liou2choi(L)))")
            warn("Diamond norm calculation did not converge.")
        end

        return 2*prob.optval
    end
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

        prob = maximize( tr( Jr*Xr + Ji*Xi ) )

        prob.constraints += tr(ρ0r) == 1
        prob.constraints += tr(ρ0i) == 0
        prob.constraints += tr(ρ1r) == 1
        prob.constraints += tr(ρ1i) == 0

        Mρ0r = reshape(M * vec(ρ0r), dy*dx, dy*dx)
        Mρ0i = reshape(M * vec(ρ0i), dy*dx, dy*dx)
        Mρ1r = reshape(M * vec(ρ1r), dy*dx, dy*dx)
        Mρ1i = reshape(M * vec(ρ1i), dy*dx, dy*dx)

        prob.constraints += isposdef( ϕ(ρ0r,ρ0i) )

        prob.constraints += isposdef( ϕ(ρ1r,ρ1i) )

        prob.constraints += isposdef( ϕ( [ Mρ0r Xr ; Xr' Mρ1r ], [ Mρ0i Xi ; -Xi' Mρ1i ] ) )

        solve!(prob, SCSSolver(verbose=0, eps=1e-6, max_iters=5_000))

        if prob.status != :Optimal
            #println("DNORM warning.")
            #println("Input: $(L)")
            #println("Input's Choi spectrum: $(eigvals(liou2choi(L)))")
            warn("Diamond norm calculation did not converge.")
        end

        return prob.optval
    end
end
