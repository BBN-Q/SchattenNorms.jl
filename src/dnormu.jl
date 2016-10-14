function orig_to_line( c1, c2 )
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
    return abs(ar - ar[mod([1:d]-2,d)+1]);
end

"""
worstfidelity(U,V) 

For two unitaries `U` and `V`, `worstfidelity(U,V)` is the worst case
fidelity between `U*a` and `V*a`, where `a` is a complex vector
with norm 1. That is, `worstfidelity(U,V)` is equal to the minimum
of `|a'*U'*V*a|^2` over all complex vectors `a` with unit norm.
"""
function worstfidelity(u::Matrix, v::Matrix)

    # some very basic error checking
    if size(u) != size(v)
        error("Input matrices do not have the same size");
    end;
  
    e  = eigvals(u'*v); # eigenvalues of u'*v
    es = sortrows(hcat(e,angle(e)+pi),by=x->x[2])
    d  = es[:,1]
    f  = 0

    es[:,2] = es[:,2]-pi
    n = length(d)
  
    # find the minimum distance from the convex hull
    # of the distinct eigenvalues to the origin. if
    # the origin is contained in the convex hull, 
    # that distance is defined as 0.
    if n==1
        f = 1
    else
        if n==2
            f = orig_to_line(d[1],d[2]);
        else 
            dn = dist_to_neighbour(angle(d));
            dn[1] = 2*pi-sum(dn[2:n])  # the sum of the angular separations is 2*pi
                                       # and the boundary cases are funny
                                       # so it is easiest to calc it this way
            dn = find(dn > pi);
            if length(dn>0)==1
                f = orig_to_line(d[dn],d[mod(dn-2,n)+1]);
            end
        end
    end
    return f^2
end

"""
dnorm(U,V)

Diamond norm distance between two unitary matrices `U` and `V`.
"""
function dnorm(U::Matrix,V::Matrix)
    return 2*sqrt(1-worstfidelity(U,V))
end
