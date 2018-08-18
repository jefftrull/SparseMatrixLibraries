# generated with seed 1529970392 and a million runs
# failure is at verifyIsApprox(A * x, b) with scalar type double
# sparse code produced this result for x:
# computed x = [           0;
#   8.44546e+16;
#   2.01104e+14;
#  -3.25312e+12;
#   -3.2318e+19]

A = [    10.875,          0,          0,          0,          0;
  -0.397597,    12.1403,          0,          0,  0.0317254;
  -0.851737, -0.0269339,    11.3113,  0.0130592,          0;
  -0.676106,          0,   0.138752,    8.57745,          0]
b = [ 10.3612;
 -2.27836;
 -10.3179;
 -7.49344]

x = A \ b

A*x

# check out suitesparse result, which seems weird
ssx = [.9527549883130686;
       -20377.936195220016;
       -49.36357536123119;
       0;
       7797911.109843667]
A*ssx
# actually it works!!! I guess matrix is underdetermined or something?

# OK let's look at the sparseqr result again using the lower precision stuff I copy-pasted
sqrx = [           0;
   -8.172e+16;
 -1.94591e+14;
  3.14778e+12;
  3.12717e+19]

A*sqrx

# Again but using the full precision outputs
sqrx = [           0;
  8.44546e+16;
  2.01104e+14;
 -3.25312e+12;
  -3.2318e+19]

A*sqrx

# both of those last two produce bad reproduction so that's not the issue

# let's try out the COLAMD support in Octave
sA = sparse(A)
caP = colamd(sA)
speye(5) (:, caP)

