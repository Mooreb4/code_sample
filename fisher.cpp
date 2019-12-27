#include "fisher.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// The purpose of this code is to produce the expected fisher information matrix defined by
// F_{i,j} = (h_{,i}|h_{,j})
// Where i,j run through parameters of the waveforms h and h_{,j} denotes the partial derivative of h with respect to the jth parameter
// and ( | ) denotes the inner product in frequency space
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

vector<vector<complex<double>>> gen_waveform_full(double M, double eta, double e0, double A, double b, double f0, double fend, double df)
{
    // Call outside code which produces the waveform in the fourier domain as a collection of harmonics
    
    // M    == Mass in solar masses
    // eta  == reduced mass ratio
    // e0   == orbital eccentricity
    // A    == overall amplitude
    // b    == Alt-theory coupling parameter
    // f0   == initial frequency
    // fend == end frequency
    // df   == fequency resolution
    
    TaylorF2e F2e(exp(M), eta, e0, exp(A), b, f0, fend, df);
    F2e.init_interps(8000);
    F2e.make_scheme();
    vector<vector<complex<double>>> vect;
    
    vect = F2e.get_F2e_min();
    
    return vect;
}
double prod_rev(vector<double> &Ai, vector<double> &Aj, vector<double> &A, vector<double> &Phii, vector<double> &Phij, vector<double> &noise, double &df)
{
    // Compute entries of the fisher matrix ((h_{,i}|h_{,j})) via method proposed in https://arxiv.org/pdf/1007.4820.pdf
    // For a SINGLE harmonic in the Fourier Response
    
    // Ai/j   == derivative of harmonic amplitude A wrt i/j
    // Phii/j == derivative of harmonic phase phi wrt i/j
    // A      == harmonic amplitude
    // noise  == detector noise
    // df     == frequency resolution
    
    int N      = Ai.size();
    double sum = 0;
    
    for(int i = 0; i < N; i++)
    {
        sum += 4./exp(noise[i])*(Ai[i]*Aj[i] + A[i]*A[i]*Phii[i]*Phij[i]);
    }
    
    return sum*df;
}

double prod_rev(vector<vector<double>> &deriv_i, vector<vector<double>> &deriv_j, vector<vector<double>> &Amps, vector<double> &noise, double &df)
{
    // Compute entries of the fisher matrix ((h_{,i}|h_{,j})) via method proposed in https://arxiv.org/pdf/1007.4820.pdf
    // For a ALL harmonics in the Fourier Response
    
    // deriv_i/j == derivatives of each harmonic wrt parameter i/j (S.T deriv_i[k] is the derivative of kth harmonic wtf i/j)
    // Amps      == overall amplitude for each harmonic (S.T Amps[k] is the amplitude of the kth harmonic)
    // noise     == detector noise
    // df        == frequency resolution
    
    int j      = deriv_i.size();
    double sum = 0;
    
    for(int k = 0; k < j; k+=2)
    {
        sum += prod_rev(deriv_i[k], deriv_j[k], Amps[k], deriv_i[k+1], deriv_j[k+1], noise, df);
    }
    
    return sum;
}

vector<vector<double>> get_amp_phs(vector<vector<complex<double>>> &harm_wav)
{
    // Get the amplitudes, A(f), and phases, phi(f), of each harmonic of the form A(f)*e^(i phi(f))
    // Returns array with rows 0... k-1 ...2 amplitudes and rows 1... k ...2 phases w/ k being the number of harmonics
    
    // harm_wav == array of harmonics, first index is nth harmonic, second is nth frequency sample
    
    int N               = harm_wav[0].size();
    int j               = harm_wav.size();
    complex<double> val = 0;
    
    vector<vector<double>> amp_phs(2*j, vector<double> (N));
    
    for(int k = 0; k < 2*j; k+=2)
    {
        for(int i = 0; i < N; i++)
        {
            val             = harm_wav[k/2][i];
            amp_phs[k][i]   = abs(val);
            amp_phs[k+1][i] = arg(val);
        }
    }
    return amp_phs;
}
vector<vector<double>> gen_amp_phs(double M, double eta, double e0, double A, double b, double f0, double fend, double df)
{
    // As above but for parameter inputs
    
    vector<vector<complex<double>>> harms = gen_waveform_full(M, eta, e0, A, b, f0, fend, df);
    vector<vector<double>> amp_phs        = get_amp_phs(harms);
    
    return amp_phs;
}

vector<double> finite_diff(vector<double> &vect_right, vector<double> &vect_left, double ep)
{
    // Simple first order finite differencing scheme
    // returns f'(x)
    
    // vect_right = f(x + ep)
    // vect_left  = f(x - ep)
    // ep         = epsilon of the FD scheme
    
    int N       = vect_right.size();
    double diff = 0;
    vector<double> deriv(N);
    
    for(int i = 0; i < N; i++)
    {
        diff = vect_right[i] - vect_left[i];
        
        // Correct for differences spanning the 2pi phase gap
        if((diff < 1) && (diff > -1))   {diff = diff;}
        else if (diff > 1)              {diff = diff - 2*M_PI;}
        else if (diff < -1)             {diff = diff + 2*M_PI;}
        
        deriv[i] = diff/(2*ep);
    }
    
    return deriv;
}

vector<vector<double>> finite_diff(vector<vector<double>> &vect_right, vector<vector<double>> &vect_left, double ep)
{
    // Finite difference scheme with vectorized input
    // returns [f_0'(x) ....., f_j'(x)]
    
    // vect_right == [f_0(x + ep) ....., f_j(x + ep)]
    // vect_left  == [f_0(x - ep) ....., f_j(x - ep)]
    // epsilon    == epsilon of the finite difference
    
    int N  = vect_right[0].size();
    int j1 = vect_right.size();
    int j2 = vect_left.size();
    int j  = 0;
    
    //Correct for maybe having a jump in harmonics on either side of the finite difference
    if (j1 > j2) {j = j2;} else {j = j1;}
    vector<vector<double>> fds (j, vector<double> (N));
    
    for(int k = 0; k < j; k++)
    {
        fds[k] = finite_diff(vect_right[k], vect_left[k], ep);
    }
    
    return fds;
}

Eigen::MatrixXd fim_ecc_BD(vector<double> &loc, vector<double> &noise, double f0, double fend, double df, double ep){
    // Compute the matrix F_{i,j} = (h_{,i}|h_{,j})
    
    // loc   == location in parameter space to get the expect fisher information matrix
    // noise == detector noise
    // f0    == lower frequency bound of integration
    // fend  == upper frequency bound of integration
    // df    == frequency discretization
    // ep    == epsilon of the finite differencing
    
    
    double M   = log(loc[0]); // Going to compute fisher information wrt to Log(Mass)
    double eta = loc[1];
    double e0  = loc[2];
    double A   = log(loc[3]); // Going to compute fisher information wrt to Log(Amplitude)
    double b   = loc[4];
    
    // First generate the waveforms required to compute the numerical derivatives (here ive just got the them as vectors with A_1, phi_1, .....)
    vector<vector<double>> M_right   = gen_amp_phs(M + ep, eta, e0, A,  b, f0, fend, df);
    vector<vector<double>> M_left    = gen_amp_phs(M - ep, eta, e0, A, b, f0, fend, df);
    
    vector<vector<double>> eta_right = gen_amp_phs(M, eta + ep, e0, A, b, f0, fend, df);
    vector<vector<double>> eta_left  = gen_amp_phs(M, eta - ep, e0, A, b, f0, fend, df);
    
    vector<vector<double>> e0_right  = gen_amp_phs(M, eta, e0 + ep, A, b, f0, fend, df);
    vector<vector<double>> e0_left   = gen_amp_phs(M, eta, e0 - ep, A, b, f0, fend, df);
    
    vector<vector<double>> b_right   = gen_amp_phs(M, eta, e0, A, b + ep, f0, fend, df);
    vector<vector<double>> b_left    = gen_amp_phs(M, eta, e0, A, b - ep, f0, fend, df);
    
    vector<vector<double>> A_right   = gen_amp_phs(M, eta, e0, A + ep, b, f0, fend, df);
    vector<vector<double>> A_left    = gen_amp_phs(M, eta, e0, A - ep, b, f0, fend, df);
    
    vector<vector<double>> A_gen     = gen_amp_phs(M, eta, e0, A, b, f0, fend, df);
    
    // Now compute the needed derivatives h_{,i}(f)
    
    vector<vector<double>> M_deriv   = finite_diff(M_right, M_left, ep);
    vector<vector<double>> eta_deriv = finite_diff(eta_right, eta_left, ep);
    vector<vector<double>> e0_deriv  = finite_diff(e0_right, e0_left, ep);
    vector<vector<double>> b_deriv   = finite_diff(b_right, b_left, ep);
    vector<vector<double>> A_deriv   = finite_diff(A_right, A_left, ep);
    
    //Now the inner products in the fisher (h_{,i}(f)|h_{,j}(f))
    double prod_mm     = prod_rev(M_deriv, M_deriv, A_gen, noise, df);
    double prod_meta   = prod_rev(M_deriv, eta_deriv, A_gen, noise, df);
    double prod_me0    = prod_rev(M_deriv, e0_deriv, A_gen, noise, df);
    double prod_mb     = prod_rev(M_deriv, b_deriv, A_gen, noise, df);
    double prod_mA     = prod_rev(M_deriv, A_deriv, A_gen, noise, df);
    double prod_etaeta = prod_rev(eta_deriv, eta_deriv, A_gen, noise, df);
    double prod_etae0  = prod_rev(eta_deriv, e0_deriv, A_gen, noise, df);
    double prod_etab   = prod_rev(eta_deriv, b_deriv, A_gen, noise, df);
    double prod_etaA   = prod_rev(eta_deriv, A_deriv, A_gen, noise, df);
    double prod_e0e0   = prod_rev(e0_deriv, e0_deriv, A_gen, noise, df);
    double prod_e0b    = prod_rev(e0_deriv, b_deriv, A_gen, noise, df);
    double prod_e0A    = prod_rev(e0_deriv, A_deriv, A_gen, noise, df);
    double prod_bb     = prod_rev(b_deriv, b_deriv, A_gen, noise, df);
    double prod_bA     = prod_rev(b_deriv, A_deriv, A_gen, noise, df);
    double prod_AA     = prod_rev(A_deriv, A_deriv, A_gen, noise, df);
    
    //Load up a fisher matrix (with added standard deviation of the uniform priors on the diagonals)
    Eigen::MatrixXd m(5,5);
    m(0,0) = prod_mm + 1.38413;
    m(1,1) = prod_etaeta + 1200.;
    m(2,2) = prod_e0e0 + 18.75;
    m(3,3) = prod_AA + 0.0017464;
    m(4,4) = prod_bb + 0.000101;
    
    m(0,1) = prod_meta;
    m(0,2) = prod_me0;
    m(0,4) = prod_mb;
    m(0,3) = prod_mA;
    m(1,0) = prod_meta;
    m(2,0) = prod_me0;
    m(4,0) = prod_mb;
    m(3,0) = prod_mA;
    
    m(1,2) = prod_etae0;
    m(1,4) = prod_etab;
    m(1,3) = prod_etaA;
    m(2,1) = prod_etae0;
    m(4,1) = prod_etab;
    m(3,1) = prod_etaA;
    
    m(2,4) = prod_e0b;
    m(2,3) = prod_e0A;
    m(4,2) = prod_e0b;
    m(3,2) = prod_e0A;
    
    m(3,4) = prod_bA;
    m(4,3) = prod_bA;
    
    return m;
}

Eigen::MatrixXd fim_BD(vector<double> &loc, vector<double> &noise, double f0, double fend, double df, double ep , double T){
    // Do an SVD decomposition of the fisher information matrix to check if the matrix is close to singular
    // if the matrix is close to singular we must modify it so it doesn't lead to numerical issues when
    // being inverted
    
    // loc   == location in parameter space to get the expect fisher information matrix
    // noise == detector noise
    // f0    == lower frequency bound of integration
    // fend  == upper frequency bound of integration
    // df    == frequency discretization
    // ep    == epsilon of the finite differencing
    // T     == Temperature of the chain in the markov chain monte carlo algorithm (MCMC)
    
    Eigen::MatrixXd m = 1./T*fim_ecc_BD(loc, noise, f0, fend, df, ep);
    
    double condition_number;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd sings(5,5);
    
    // If I don't explicitly set the off-diagonals to 0 there are sometimes strange issues
    for(int i = 0; i < 5; i++)
    {
        for int(j = 0; j < 5; j++)
        {
            if(i != j)
            {
                sings(i, j) = 0;
            } else {
                sings(i, i) = svd.singularValues()(i);
            }
        }
    }
    
    // Condition number to be checked for singularity
    condition_number = sings(0,0)/sings(4,4);
    while(condition_number > 5*1e6)
    {
        // Increase the value leading to singularity and recompute SVD until no longer singular
        sings(4,4) *= 1.1;
        m          = svd.matrixU()*sings*svd.matrixV().adjoint();
        
        svd.compute(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
        
        for(int i = 0; i < 5; i++)
        {
            sings(i, i) = svd.singularValues()(i);
        }
        
        condition_number = sings(0,0)/sings(4,4);
    }
    return m;
}

void fisher_prop_ecc_BD(vector<double> &loc, vector<double> &prop, Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es, const gsl_rng * r){
    //Make a jump based on the eigenvalues and eigenvectors of the Fisher information matrix
    
    // loc  == current location of the MCMC chain
    // prop == proposal location contain of the MCMC chain (to be filled)
    // es   == Eigensystem of the Fisher information matrix
    // r    == ptr to the location of the random number generator
    
    
    double delt     = gsl_ran_gaussian (r, 1.);
    int i           = floor(gsl_ran_flat(r, 0, 5));
    if (i == 5){i = 4;}
    
    prop[0] = loc[0]*exp(delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(0));
    prop[1] = loc[1] + delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(1);
    prop[2] = loc[2] + delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(2);
    prop[3] = loc[3]*exp(delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(3));
    prop[4] = loc[4] + delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(4);
    
}
