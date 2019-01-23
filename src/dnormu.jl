import LinearAlgebra: eigvals

function orig_to_line( c1::Number, c2::Number )
    if isapprox(c1,c2)
        return abs(c1)
    else
        x1 = real(c1);
        y1 = imag(c1);
        x2 = real(c2);
        y2 = imag(c2);
        return abs((x2-x1)*y1-x1*(y2-y1))/sqrt((x2-x1)^2+(y2-y1)^2);
    end
end

function dist_to_neighbour( ar )
    d = length(ar);
    #return abs(ar - ar[mod((0:d-1)-1,d)+1]);
    neighbour_index = broadcast(mod, (1:d) .-2, d) .+ 1
    return abs.(ar - ar[neighbour_index]);
end

"""
worstfidelity(U,V)

For two unitaries `U` and `V`, `worstfidelity(U,V)` is the worst case
Jozsa fidelity between `U*a` and `V*a`, where `a` is a complex vector
with norm 1. That is, `worstfidelity(U,V)` is equal to the minimum
of `|a'*U'*V*a|^2` over all complex vectors `a` with unit norm.
"""
function worstfidelity(u::Matrix, v::Matrix)
    if size(u) != size(v)
        error("Input matrices do not have the same size")
    end

    #local variables
    e  = eigvals(u'*v) # eigenvalues of u'*v
    shifted_e = map(x -> angle(x) + pi, e)
    es = sortslices(hcat(e,shifted_e),dims=1,by=x->real(x[2]))
    d  = es[:,1]
    f  = 0

    #es(:,2) = es(:,2)-pi;
    n = length(d)

    # find the minimum distance from the convex hull
    # of the distinct eigenvalues to the origin. if
    # the origin is contained in the convex hull,
    # that distance is defined as 0.
    if n==1
        f = 1
    else
        if n==2
            f = orig_to_line(d[1],d[2])
        else
            dn = dist_to_neighbour(angle.(d))
            dn[1] = 2*pi-sum(dn[2:n])  # the sum of the angular separations is 2*pi
            # and the boundary cases are funny
            # so it is best to calc it this way
            dn = findall(dn .> pi)
            if length(dn)==1
                f = orig_to_line(d[dn[1]],d[mod(dn[1]-2,n)+1])
            end
        end
    end
    return abs(f)^2
end

"""
ddist(U,V)

Diamond norm distance between two linear CPTP superoperators.
  Equivalent to dnrom(E-F), but under the assumption that `E` and `F`
  are CPTP supeoperators, but the `ddist` call is
  much more accurate and faster due to properties of CPTP
  superoperators.

See `ddistu` for a similar implementation of `dnorm` customized
to the difference between unitary operations.

"""
ddist(E::Matrix,F::Matrix) = dnormcptp(E,F)

"""
ddist(U,V)

Diamond norm distance between two unitary operations.
  Equivalent to `dnorm(liou(U)-liou(V))`, under the assumption `U` and
  `V` are unitary matrices. However the `ddist` call is much more
  accurate and much faster due to properties of unitary matrices and
  the corresponding superoperators.

  **Note:** for this particular case, the matrices in question are not
    the superoperators corresponding to the unitary operation, but
    rather the unitary operations themselves.

See `ddist` for a similar implementation of `dnorm` customized
to the difference between CPTP maps.

"""
function ddistu(U::Matrix,V::Matrix)
    w = worstfidelity(U,V)
    return 2*sqrt(1-w)
end
