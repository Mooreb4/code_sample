#include <vector>
#include <iostream>
#include <math.h>
#include <string>
#include <iomanip>
#include <stdio.h>
#include <fstream>
#include <ostream>
#include <gsl/gsl_randist.h>

#include "llhood_maxd.hpp"
#include "fisher.hpp"

unsigned int num_params = 5;
unsigned int num_diff_evol_samples = 1000;
double f_begin = 0;
double f_end = 1000;
double df = 0.015625;

struct Chain_BD{
    double temp;
    double loglike_loc;
    double loglike_prop;
    double hast_ratio;
    double urv;
    int count_in_temp;
    int count_swap;
    int count_in_temp_accpt;
    int count_swap_accpt;
    vector<double> loc;
    vector<double> prop;
    Eigen::MatrixXd fisher;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_sys;
    vector<vector<double>> DE_samples;
    int DE_track;
    
    Chain_BD(double M, double eta, double e0, double A, double b, double T, double fend,
             double df_fish, double ep_fish, vector<double> &noise, vector<double> &noise_fish,
             vector<complex<double>> &h2) :
    
              temp(T), count_in_temp(0), count_swap(0), count_in_temp_accpt(0),
              count_swap_accpt(0), loglike_prop(0), hast_ratio(0), urv(0)
    {
        DE_track = 0;
        DE_samples.resize(num_diff_evol_samples, vector<double> (num_params));
        loc.resize(num_params);
        prop.resize(num_params);
        loc[0] = M;
        loc[1] = eta;
        loc[2] = e0;
        loc[3] = A;
        loc[4] = b;
        loglike_loc = loglike(loc, f_begin, fend, df, h2, noise, T);
        fisher = fim_BD(loc, noise_fish, f_begin, fend, df_fish, ep_fish, T, 3);
        eigen_sys.compute(fisher);
    }
};

struct Chain_GR{
    double temp;
    double loglike_loc;
    double loglike_prop;
    double hast_ratio;
    double urv;
    int count_in_temp;
    int count_swap;
    int count_in_temp_accpt;
    int count_swap_accpt;
    vector<double> loc;
    vector<double> prop;
    Eigen::MatrixXd fisher;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_sys;
    vector<vector<double>> DE_samples;
    int DE_track;
    
    Chain_GR(double M, double eta, double e0, double A, double b, double T, double fend,
             double df_fish, double ep_fish, vector<double> &noise, vector<double> &noise_fish,
             vector<complex<double>> &h2) :
    
              temp(T), count_in_temp(0), count_swap(0), count_in_temp_accpt(0),
              count_swap_accpt(0), loglike_prop(0), hast_ratio(0), urv(0)
    {
        DE_track = 0;
        DE_samples.resize(num_diff_evol_samples, vector<double> (num_params));
        loc.resize(num_params);
        prop.resize(num_params);
        loc[0] = M;
        loc[1] = eta;
        loc[2] = e0;
        loc[3] = A;
        loc[4] = b;
        loglike_loc = loglike(loc, f_begin, fend, df, h2, noise, T);
        fisher = fim_GR(loc, noise_fish, f_begin, fend, df_fish, ep_fish, T, 3);
        eigen_sys.compute(fisher);
    }
};

struct Chain{
    int gr_true;
    int count_TD_GR_to_BD = 0;
    int count_TD_GR_to_BD_accpt = 0;
    int count_TD_BD_to_GR = 0;
    int count_TD_BD_to_GR_accpt = 0;
    
    int interchain_count_TD_GR_to_BD = 0;
    int interchain_count_TD_GR_to_BD_accpt = 0;
    int interchain_count_TD_BD_to_GR = 0;
    int interchain_count_TD_BD_to_GR_accpt = 0;
    
    Chain_GR c_GR;
    Chain_BD c_BD;
    
    Chain(double M, double eta, double e0, double A, double b, double T, double fend,
          double df_fish, double ep_fish, vector<double> &noise, vector<double> &noise_fish,
          vector<complex<double>> &h2):
    
            c_GR(M, eta, e0, A, 0, T, fend, df_fish, ep_fish, noise, noise_fish, h2),
            c_BD(M, eta, e0, A, b, T, fend, df_fish, ep_fish, noise, noise_fish, h2)
    {
        gr_true = 0;
    }
};

// Make "ep_fisher" a global variable and stop handing it around

// after the functions have been made to take mostly chains they can likely be turned into member functions and lumped into a class.
// compile before that

vector<complex<double>> gen_waveform(double M, double eta, double e0, double A, double b, double fend);
vector<complex<double>> sum_f2(vector<vector<complex<double>>> &vect);
void jump(Chain &c, const gsl_rng * r, double fend, vector<complex<double>> &h2, vector<double> &noise);
void jump_TD_GR_to_BD(Chain &c, const gsl_rng * r, double fend, vector<complex<double>> &h2, vector<double> &noise);
void jump_TD_BD_to_GR(Chain &c, const gsl_rng * r, double fend, vector<complex<double>> &h2, vector<double> &noise);
double eval_log_g_gaus (double x, double mu, double sigma);
void jump_GR(Chain &c, const gsl_rng * r, double fend, vector<complex<double>> &h2, vector<double> &noise);
void set_loc(vector<double> &prop, vector<double> &loc);
// These DE_props can be made to just take the chains
void DE_prop_GR(vector<double> &loc, vector<double> &prop, const gsl_rng * r, int N_DE_samples, vector<vector<double>> &DE_samples);
void jump_BD(Chain &c, const gsl_rng * r, double fend, vector<complex<double>> &h2, vector<double> &noise);
void DE_prop_BD(vector<double> &loc, vector<double> &prop, const gsl_rng * r, int N_DE_samples, vector<vector<double>> &DE_samples);
void prior_prop_GR(vector<double> &prop, const gsl_rng * r);
void prior_prop_BD(vector<double> &prop, const gsl_rng * r);
int check_priors_GR(vector<double> &prop);
int check_priors_BD(vector<double> &prop);
void update_fisher(Chain &c, double fend, double df_fish, vector<double> &noise_fish, double ep_fish);
void update_fisher_GR(Chain &c, double fend, double df_fish, vector<double> &noise_fish, double ep_fish);
void update_fisher_BD(Chain &c, double fend, double df_fish, vector<double> &noise_fish, double ep_fish);
void write_to_DE(Chain &c);
void write_to_DE(Chain_GR &c);
void write_to_DE(Chain_BD &c);
void inter_chain_swap(Chain &c1, Chain &c2, const gsl_rng * r, double fend, double df, vector<complex<double>> &h2, vector<double> &noise);
void inter_chain_swap_trans(Chain &c1, Chain &c2, const gsl_rng * r, double fend, double df, vector<complex<double>> &h2, vector<double> &noise);
void inter_chain_swap_same_dim(Chain_GR &c1, Chain_GR &c2, const gsl_rng * r);
void inter_chain_swap_same_dim(Chain_BD &c1, Chain_BD &c2, const gsl_rng * r);
void swap_fishers(Eigen::MatrixXd &fish1, Eigen::MatrixXd &fish2, Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> &es1 , Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> &es2, double T1, double T2);
void swap_loc(vector<double> &loc1, vector<double> &loc2);
void record(Chain &c, vector<vector<vector<double>>> &chain_store, vector<vector<double>> &like_store, int chain_num, int i);
void write_vec_to_vec(vector<vector<double>> &samples, vector<double> &sample, int i);
void write_vec_to_file(vector<vector<double>> &vect, string filename, string path);
void write_vec_to_file(vector<double> &vect, string str, string path);
void write_vec_to_file(vector<vector<double>> &vect, string filename, string path, ofstream &out, int start, int end);
void write_vec_to_file(vector<double> &vect, string filename, string path, ofstream &out, int start, int end);
void cout_chain_info(Chain c);
void cout_chain_info(Chain_GR c);
void cout_chain_info(Chain_BD c);
void cout_vec(vector<double> &vec);
void print_accpt_ratios(vector<Chain> &c_vect);
void print_accpt_ratios(Chain &c, int i);

int main (int argc, const char * argv[]){
    
    auto M_in     = stod(argv[1]);
    auto eta_in   = stod(argv[2]);
    auto e0_in    = stod(argv[3])/1000.;
    auto A_in     = exp(stod(argv[4]));
    auto b_exp    = stod(argv[5])/100.;
    auto b_in     = pow(10, -b_exp);
    auto SNR_in   = stod(argv[6]);
    
    if (e0_in == 0){e0_in = 0.000001;}
    
    std::cout << "Running e = " << e0_in << " and -log10(b) = " << b_exp << std::endl;
    
    ////////////////////////////////////////////////////////
    // Write in data file (Injection)
    ////////////////////////////////////////////////////////
    
    std::cout << "Injection Begun" << std::endl;
    
    auto h2  = gen_waveform(M_in, eta_in, e0_in, A_in, b_in, f_begin, f_end, df);
    auto data_size               = h2.size();
    
    vector<double> freqs(data_size);
    
    for(int i = 0; i < data_size; i++)
    {
        freqs[i] = df*i;
    }
    
    auto fend = freqs[data_size - 1];
    
    std::cout << "Injection Complete" << std::endl;
    
    ////////////////////////////////////////////////////////
    //     Set up noise for likelihood evaluations
    //     Set up downsampled noise for fisher
    ////////////////////////////////////////////////////////
    
    auto j         = 0;
    auto num_pts   = 3000;
    
    double f_val[num_pts];
    double noise_val[num_pts];
    double in_f;
    double in_noise;
    
    auto f_noise_low  = 1.0;
    auto f_noise_high = 4096.0;
    
    ifstream noisedat ("AdLIGODwyer.dat");
    while(noisedat >> in_f >> in_noise)
    {
        f_val[j] = in_f;
        noise_val[j] = log(in_noise*in_noise);
        j++;
    }
    
    gsl_interp_accel *acc         = gsl_interp_accel_alloc ();
    gsl_spline *noise_spline      = gsl_spline_alloc (gsl_interp_cspline, num_pts);
    gsl_spline_init (noise_spline, f_val, noise_val, num_pts);
    
    //noise for the likelihood
    vector<double> noise(data_size);
    for (int i = 0; i < data_size; i++)
    {
        if (freqs[i] > f_noise_low &&  freqs[i] < f_noise_high )
        {
            noise[i] = gsl_spline_eval (noise_spline, freqs[i], acc);
        }
        else
        {
            noise[i] = pow(10,10); //effectively make the noise infinite below 1Hz
        }
    }
    
    //downsampled noise for the fisher
    auto df_fish = 0.25;
    auto ep_fish = 1e-8;
    
    auto N_down_noise = std::floor(fend/df_fish) + 1;
    
    vector<double> noise_fish(N_down_noise);
    double f = 0;
    for (int i = 0; i < N_down_noise; i++)
    {
        f = df_fish*i;
        
        if (f > f_noise_low &&  f < f_noise_high )
        {
            noise_fish[i] = gsl_spline_eval (noise_spline, f, acc);
        }
        else
        {
            noise_fish[i] = pow(10,10); //effectively make the noise infinite below 1Hz
        }
    }
    
    ////////////////////////////////////////////////////////
    // Root find the A (amplitude) to make the SNR a selected value
    // Uses secant method
    ////////////////////////////////////////////////////////
    
    std::cout << "Beginning root finding" << std::endl;
    
    double A_prev = A_in;
    double A_curr = A_in - A_in/100;
    
    vector<complex<double>> h1 = gen_waveform(M_in, eta_in, e0_in, A_prev, b_in, f_begin, fend, df);
    h2 = gen_waveform(M_in, eta_in, e0_in, A_curr, b_in, f_begin, fend, df);
    
    double snr_shoot_1      = sqrt(get_snr_sq(h1, noise, df));
    double snr_shoot_2      = sqrt(get_snr_sq(h2, noise, df));
    double snr_diff_prev    = snr_shoot_1 - SNR_in;
    double snr_diff_curr    = snr_shoot_2 - SNR_in;
    double A_next           = A_prev - snr_diff_prev*(A_prev-A_curr)/(snr_diff_prev-snr_diff_curr);

    while(abs(snr_diff_curr) > 0.001)
    {
        A_prev              = A_curr;
        snr_diff_prev       = snr_diff_curr;
        A_curr              = A_next;
        h2                  = gen_waveform(M_in, eta_in, e0_in, A_curr, b_in, f_begin, fend, df);
        snr_shoot_2         = sqrt(get_snr_sq(h2, noise, df));
        snr_diff_curr       = snr_shoot_2 - SNR_in;
        A_next              = A_prev - snr_diff_prev*(A_prev-A_curr)/(snr_diff_prev-snr_diff_curr);
        
        std::cout << "Amp = " << A_curr << " Cond = " << snr_diff_curr << std::endl;
    }
    
    cout << "The Final Injected Parameters are M  = " << M_in << " Eta = " << " eta_in " << eta_in << " e_ref = " << e0_in << " A = " << A_curr << endl;
    
    ////////////////////////////////////////////////////////
    // Initialize the chains
    ////////////////////////////////////////////////////////
    
    std::cout << "Initializing Chains" << std::endl;
    
    int N_chain          = stoi(argv[7]);
    double temp_spacing  = stod(argv[8]);
    
    vector<Chain> chains;
    
    //initialize many chains
    for(int i = 0; i < N_chain; i++)
    {
        Chain c(M_in, eta_in, e0_in, A_curr, b_in, 1*pow(temp_spacing, i), fend, df_fish, ep_fish, noise, noise_fish, h2);
        chains.push_back(c);
    }
    
    std::cout << "Chains inititialized" << std::endl;
    ////////////////////////////////////////////////////////
    // MCMC Routine
    ////////////////////////////////////////////////////////
    
    ofstream output;
    
    int N_jumps         = stoi(argv[9]);
    const gsl_rng * r   = gsl_rng_alloc (gsl_rng_taus);
    
    vector<vector<vector<double>>> chain_store(N_chain, vector<vector<double>>(N_jumps, vector<double>(num_params)));
    vector<vector<double>> like_store(N_chain, vector<double>(N_jumps));
    
    for(int i = 0; i < N_jumps; i++)
    {
        if( i % 5 != 0)
        { //Within Tempurature jumps
            for(int j = 0; j < N_chain; j++)
            {
                jump(chains[j], r, fend, h2 ,noise);
            }
        }
        else
        { //Interchain jumps
            for(int j = 0; j < N_chain - 1; j++)
            {
                inter_chain_swap(chains[j], chains[j+1], r, fend, df, h2, noise);
            }
        }
        //Record Likelihood and Samples.
        for(int j = 0; j < N_chain; j++)
        {
            record(chains[j], chain_store, like_store, j, i);
        }
        //Write to differential evolution list every 100 jumps
        if ( i % 100 == 0)
        {
            for(int j = 0; j < N_chain; j++)
            {
                write_to_DE(chains[j]);
            }
        }
        if (i % 10000 == 0)
        {
            for(int j = 0; j < N_chain; j++)
            {
                cout << "MCMC step = " << i << endl;
                cout_chain_info(chains[j]);
            }
        }
        //Periodically update Fishers.
        if( i % 800 == 0 )
        {
            for(int j = 0; j < N_chain; j++)
            {
                update_fisher(chains[j], fend, df_fish, noise_fish, ep_fish);
            }
        }
        //Periodically print acceptance ratios
        if ( i % 10000 == 0)
        {
            print_accpt_ratios(chains);
        }
        //Periodically write data to file
        if ( i % 20000 == 0 && i > 0)
        {
            write_vec_to_file(chain_store[0], "Samples_N_"+to_string(N_jumps)+"_chain_"+to_string(0)+"_Mc_"+to_string(M_in)+"e_ref"+to_string(e0_in)+"_SNR_"+to_string(SNR_in)+"_b_"+to_string(b_exp)+".txt", "/samples", output, i - 20000, i);
            write_vec_to_file(like_store[0], "likelihood_N_"+to_string(N_jumps)+"_chain_"+to_string(0)+"_Mc_"+to_string(M_in)+"e_ref"+to_string(e0_in)+"_SNR_"+to_string(SNR_in)+"_b_"+to_string(b_exp)+".txt", "/likelihoods", output, i - 20000, i);
        }
    }
    
    //Write finished samples to file
    write_vec_to_file(chain_store[0], "Samples_N_"+to_string(N_jumps)+"_chain_"+to_string(0)+"_Mc_"+to_string(M_in)+"e_ref"+to_string(e0_in)+"_SNR_"+to_string(SNR_in)+"_b_"+to_string(b_exp)+".txt", "/samples");
    write_vec_to_file(like_store[0], "likelihood_N_"+to_string(N_jumps)+"_chain_"+to_string(0)+"_Mc_"+to_string(M_in)+"e_ref"+to_string(e0_in)+"_SNR_"+to_string(SNR_in)+"_b_"+to_string(b_exp)+".txt", "/likelihoods");

    return 0;
}

vector<complex<double>> gen_waveform(double M, double eta, double e0, double A, double b, double fend){
    TaylorF2e F2e(M, eta, e0, A, b, f_begin, fend, df);
    F2e.init_interps(1000);
    F2e.make_scheme();
    vector<vector<complex<double>>> vect;
    vect = F2e.get_F2e_min();
    vector<complex<double>> summedf2 = sum_f2(vect);
    
    return summedf2;
}
vector<complex<double>> sum_f2(vector<vector<complex<double>>> &vect){
    int N = vect[0].size();
    int j = vect.size();
    vector<complex<double>> summed(N);
    
    for (int i = 0; i < N; i++)
    {
        for(int k = 0; k < j; k++)
        {
            summed[i] += vect[k][i];
        }
    }
    
    return summed;
}

void jump(Chain &c, const gsl_rng * r, double fend, vector<complex<double>> &h2, vector<double> &noise){
    double uniform_rv = gsl_ran_flat(r, 0, 1);
    
    if(uniform_rv > 0.1)
    {
        // Do a within dimension jump
        if (c.gr_true == 1)
        {
            // A GR Jump
            jump_GR(c, r, fend, h2, noise);
            c.c_GR.count_in_temp++;
        }
        else
        {
            // A BD Jump
            jump_BD(c, r, fend, h2, noise);
            c.c_BD.count_in_temp++;
        }
    }
    else
    {
        // Do a within temp transdimensional jump
        if (c.gr_true == 1)
        {
            //Jump GR to BD
            jump_TD_GR_to_BD(c, r, fend, h2, noise);
        }
        else
        {
            jump_TD_BD_to_GR(c, r, fend, h2, noise);
        }
    }
}

void jump_TD_GR_to_BD(Chain &c, const gsl_rng * r, double fend, vector<complex<double>> &h2, vector<double> &noise){
    // propose jump
    c.c_BD.prop[0] = c.c_GR.loc[0];
    c.c_BD.prop[1] = c.c_GR.loc[1];
    c.c_BD.prop[2] = c.c_GR.loc[2];
    c.c_BD.prop[3] = c.c_GR.loc[3];
    c.c_BD.prop[4] = exp(gsl_ran_flat(r, -23.0259, -4.60517));
    
    //The value of ln(p(b)) from the draw and likelihood
    double pb       = log(1./c.c_BD.prop[4]*0.0542868);
    int check_pri   = check_priors_BD(c.c_BD.prop);
    
    if(check_pri != 1)
    {
        double urn          = gsl_ran_flat(r,0,1.);
        c.c_BD.loglike_prop = loglike(c.c_BD.prop, f_begin, fend, df, h2, noise, c.c_BD.temp);
        c.c_BD.hast_ratio   = min(1., 2825.16*exp(c.c_BD.loglike_prop - c.c_GR.loglike_loc - pb));
        
        if(c.c_BD.hast_ratio >= urn)
        {
            set_loc(c.c_BD.prop, c.c_BD.loc);
            c.c_BD.loglike_loc = c.c_BD.loglike_prop;
            c.count_TD_GR_to_BD_accpt++;
            c.gr_true           = 0;
        }
    }
    c.count_TD_GR_to_BD++;
}

void jump_TD_BD_to_GR(Chain &c, const gsl_rng * r, double fend, vector<complex<double>> &h2, vector<double> &noise){
    // propose jump
    c.c_GR.prop[0] = c.c_BD.loc[0];
    c.c_GR.prop[1] = c.c_BD.loc[1];
    c.c_GR.prop[2] = c.c_BD.loc[2];
    c.c_GR.prop[3] = c.c_BD.loc[3];
    c.c_GR.prop[4] = 0;
    
    //The value of ln(p(b)) from the draw and likelihood
    double pb       = log(1./c.c_BD.loc[4]*0.0542868);
    int check_pri   = check_priors_GR(c.c_GR.prop);
    
    if(check_pri != 1)
    {
        double urn          = gsl_ran_flat(r,0,1.);
        c.c_GR.loglike_prop = loglike(c.c_GR.prop, 0, fend, df, h2, noise, c.c_GR.temp);
        c.c_GR.hast_ratio   = min(1.,0.0003539626*exp(c.c_GR.loglike_prop - c.c_BD.loglike_loc + pb));
        
        if(c.c_GR.hast_ratio >= urn)
        {
            set_loc(c.c_GR.prop, c.c_GR.loc);
            c.c_GR.loglike_loc = c.c_GR.loglike_prop;
            c.count_TD_BD_to_GR_accpt++;
            c.gr_true = 1;
        }
    }
    c.count_TD_BD_to_GR++;
}

double eval_log_g_gaus (double x, double mu, double sigma)
{
    double prefac = log(1./(pow(2*M_PI,1./2.)*sigma));
    double expfac = -(x - mu)*(x - mu)/(2*sigma*sigma);
    
    return prefac + expfac;
}

void jump_GR(Chain &c, const gsl_rng * r, double fend, vector<complex<double>> &h2, vector<double> &noise)
{
    double jump_roll = gsl_ran_flat(r, 0, 1);
    
    if(jump_roll > 0.2)
    {
        fisher_prop_ecc_GR(c.c_GR.loc, c.c_GR.prop, c.c_GR.eigen_sys, r);
    }
    else if (jump_roll > 0.05)
    {
        if(c.c_GR.DE_track < 2) {fisher_prop_ecc_GR(c.c_GR.loc, c.c_GR.prop, c.c_GR.eigen_sys, r);}
        else {DE_prop_GR(c.c_GR.loc, c.c_GR.prop, r, c.c_GR.DE_track, c.c_GR.DE_samples);}
    }
    else
    {
        prior_prop_GR(c.c_GR.prop, r);
    }
    
    int check_pri = check_priors_GR(c.c_GR.prop);
    
    if(check_pri != 1)
    {
        double urn             = gsl_ran_flat(r, 0, 1.);
        c.c_GR.loglike_prop    = loglike(c.c_GR.prop, f_begin, fend, df, h2, noise, c.c_GR.temp);
        c.c_GR.hast_ratio      = min(1., exp(c.c_GR.loglike_prop - c.c_GR.loglike_loc));
        
        if(c.c_GR.hast_ratio >= urn)
        {
            set_loc(c.c_GR.prop, c.c_GR.loc);
            c.c_GR.loglike_loc = c.c_GR.loglike_prop;
            c.c_GR.count_in_temp_accpt;
        }
    }
}

void set_loc(vector<double> &prop, vector<double> &loc){
    loc[0] = prop[0];
    loc[1] = prop[1];
    loc[2] = prop[2];
    loc[3] = prop[3];
    loc[4] = prop[4];
}

void DE_prop_GR(vector<double> &loc, vector<double> &prop, const gsl_rng * r, int N_DE_samples, vector<vector<double>> &DE_samples){
    double fact = gsl_ran_gaussian (r, 0.751319);
    int i       = floor(gsl_ran_flat(r, 0, N_DE_samples - 1));
    int j       = floor(gsl_ran_flat(r, 0, N_DE_samples - 1));
    prop[0]     = exp(log(loc[0]) + fact*(log(DE_samples[i][0]) - log(DE_samples[j][0])));
    prop[1]     = loc[1] + fact*(DE_samples[i][1] - DE_samples[j][1]);
    prop[2]     = loc[2] + fact*(DE_samples[i][2] - DE_samples[j][2]);
    prop[3]     = exp(log(loc[3]) + fact*(log(DE_samples[i][3]) - log(DE_samples[j][3])));
}

void jump_BD(Chain &c, const gsl_rng * r, double fend, vector<complex<double>> &h2, vector<double> &noise)
{
    double jump_roll = gsl_ran_flat(r, 0, 1);
    
    if(jump_roll > 0.2)
    {
        fisher_prop_ecc_BD(c.c_BD.loc, c.c_BD.prop, c.c_BD.eigen_sys, r);
    }
    else if (jump_roll > 0.05)
    {
        if(c.c_BD.DE_track < 2) {fisher_prop_ecc_BD(c.c_BD.loc, c.c_BD.prop, c.c_BD.eigen_sys, r);}
        else {DE_prop_BD(c.c_BD.loc, c.c_BD.prop, r, c.c_BD.DE_track, c.c_BD.DE_samples);}
    }
    else
    {
        prior_prop_BD(c.c_BD.prop, r);
    }
    
    int check_pri = check_priors_BD(c.c_BD.prop);
    
    if(check_pri != 1)
    {
        double urn             = gsl_ran_flat(r, 0, 1.);
        c.c_BD.loglike_prop    = loglike(c.c_BD.prop, f_begin, fend, df, h2, noise, c.c_BD.temp);
        c.c_BD.hast_ratio      = min(1., exp(c.c_BD.loglike_prop - c.c_BD.loglike_loc));
        if(c.c_BD.hast_ratio >= urn)
        {
            set_loc(c.c_BD.prop, c.c_BD.loc);
            c.c_BD.loglike_loc = c.c_BD.hast_ratio;
            c.c_BD.count_in_temp_accpt++;
        }
    }
}

void DE_prop_BD(vector<double> &loc, vector<double> &prop, const gsl_rng * r, int N_DE_samples, vector<vector<double>> &DE_samples)
{
    double fact = gsl_ran_gaussian (r, 0.632456);
    int i       = floor(gsl_ran_flat(r, 0, N_DE_samples - 1));
    int j       = floor(gsl_ran_flat(r, 0, N_DE_samples - 1));
    
    prop[0] = exp(log(loc[0]) + fact*(log(DE_samples[i][0]) - log(DE_samples[j][0])));
    prop[1] = loc[1] + fact*(DE_samples[i][1] - DE_samples[j][1]);
    prop[2] = loc[2] + fact*(DE_samples[i][2] - DE_samples[j][2]);
    prop[3] = exp(log(loc[3]) + fact*(log(DE_samples[i][3]) - log(DE_samples[j][3])));
    prop[4] = loc[4] + fact*(DE_samples[i][4] - DE_samples[j][4]);
}

void prior_prop_GR(vector<double> &prop, const gsl_rng * r)
{
    prop[0] = exp(gsl_ran_flat(r, -0.223144, 2.30259));
    prop[1] = gsl_ran_flat(r, 0.08, 0.25);
    prop[2] = gsl_ran_flat(r, 0.000001, 0.805);
    prop[3] = exp(gsl_ran_flat(r, -46.0517, -36.8414));
}

void prior_prop_BD(vector<double> &prop, const gsl_rng * r)
{
    prop[0] = exp(gsl_ran_flat(r, -0.223144, 2.30259));
    prop[1] = gsl_ran_flat(r, 0.08, 0.25);
    prop[2] = gsl_ran_flat(r, 0.000001, 0.805);
    prop[3] = exp(gsl_ran_flat(r, -46.0517, -36.8414));
    prop[4] = gsl_ran_flat(r, 0.0, 0.0003539626);
}

int check_priors_GR(vector<double> &prop)
{
    int cont = 0;
    
    if(prop[0] > 10     || prop[0] < 0.8)       {cont = 1;}
    if(prop[1] > 0.25   || prop[1] < 0.08)      {cont = 1;}
    if(prop[2] > 0.805  || prop[2] < 0.000001)  {cont = 1;}
    if(prop[3] > 1e-16  || prop[3] < 1e-20)     {cont = 1;}
    
    return cont;
}
int check_priors_BD(vector<double> &prop)
{
    int cont = 0;
    
    if(prop[0] > 10           || prop[0] < 0.8)         {cont = 1;}
    if(prop[1] > 0.25         || prop[1] < 0.08)        {cont = 1;}
    if(prop[2] > 0.805        || prop[2] < 0.000001)    {cont = 1;}
    if(prop[3] > 1e-16        || prop[3] < 1e-20)       {cont = 1;}
    if(prop[4] > 0.0003539626 || prop[4] < 0.0)         {cont = 1;}
    
    return cont;
}

void update_fisher(Chain &c, double fend, double df_fish, vector<double> &noise_fish, double ep_fish)
{
    if(c.gr_true == 1)
    {
        update_fisher_GR(c, fend, df_fish, noise_fish, ep_fish);
    }
    else
    {
        update_fisher_BD(c, fend, df_fish, noise_fish, ep_fish);
    }
}

void update_fisher_GR(Chain &c, double fend, double df_fish, vector<double> &noise_fish, double ep_fish)
{
    c.c_GR.fisher = fim_GR(c.c_GR.loc, noise_fish, f_begin, fend, df_fish, ep_fish, c.c_GR.temp, 3);
    c.c_GR.eigen_sys.compute(c.c_GR.fisher);
}

void update_fisher_BD(Chain &c, double fend, double df_fish, vector<double> &noise_fish, double ep_fish)
{
    c.c_BD.fisher = fim_BD(c.c_BD.loc, noise_fish, 0, fend, df_fish, ep_fish, c.c_BD.temp, 3);
    c.c_BD.eigen_sys.compute(c.c_GR.fisher);
}

void write_to_DE(Chain &c)
{
    if(c.gr_true == 1)
    {
        write_to_DE(c.c_GR);
    } else
    {
        write_to_DE(c.c_BD);
    }
}

void write_to_DE(Chain_BD &c)
{
    if(c.DE_track == 1000)
    {
        c.DE_track = 0;
    }
    c.DE_samples[c.DE_track][0] = c.loc[0];
    c.DE_samples[c.DE_track][1] = c.loc[1];
    c.DE_samples[c.DE_track][2] = c.loc[2];
    c.DE_samples[c.DE_track][3] = c.loc[3];
    c.DE_samples[c.DE_track][4] = c.loc[4];
    c.DE_track++;
}

void write_to_DE(Chain_GR &c)
{
    if(c.DE_track == 1000)
    {
        c.DE_track = 0;
    }
    c.DE_samples[c.DE_track][0] = c.loc[0];
    c.DE_samples[c.DE_track][1] = c.loc[1];
    c.DE_samples[c.DE_track][2] = c.loc[2];
    c.DE_samples[c.DE_track][3] = c.loc[3];
    c.DE_samples[c.DE_track][4] = c.loc[4];
    c.DE_track++;
}

void inter_chain_swap(Chain &c1, Chain &c2, const gsl_rng * r, double fend, double df, vector<complex<double>> &h2, vector<double> &noise)
{
    if(c1.gr_true == 1 && c2.gr_true == 1)
    {
        inter_chain_swap_same_dim(c1.c_GR, c2.c_GR, r);
        c1.c_GR.count_swap++;
    }
    else if (c1.gr_true == 0 && c2.gr_true == 0)
    {
        inter_chain_swap_same_dim(c1.c_BD, c2.c_BD, r);
        c1.c_BD.count_swap++;
    }
    else
    {
        inter_chain_swap_trans(c1, c2, r, fend, df, h2, noise);
    }
    
}

void inter_chain_swap_trans(Chain &c1, Chain &c2, const gsl_rng * r, double fend, double df, vector<complex<double>> &h2, vector<double> &noise){
    double T1 = c1.c_GR.temp;
    double T2 = c2.c_GR.temp;
    
    if(c1.gr_true == 1 &&  c2.gr_true == 0)
    { // swap is up from BD to GR
        double likeT1X2 = c2.c_BD.loglike_loc*T2/T1; //likelihood of location 2 at temp 1
        double likeT2X1 = c1.c_GR.loglike_loc*T1/T2; //likelihood of location 1 at temp 2
        double rh       = min(1., exp((likeT1X2 + likeT2X1)-(c1.c_GR.loglike_loc + c2.c_BD.loglike_loc)));
        double urn      = gsl_ran_flat(r, 0, 1.);
        
        if(rh >= urn)
        {
            set_loc(c2.c_BD.loc, c1.c_BD.loc); // sets chain1's BD walker to that of chain2's
            set_loc(c1.c_GR.loc, c2.c_GR.loc); // sets chain2's GR walker to that of chain1's
            c1.c_BD.loglike_loc = likeT1X2;
            c2.c_GR.loglike_loc = likeT2X1;
            c1.gr_true = 0;
            c2.gr_true = 1;
            c1.interchain_count_TD_BD_to_GR_accpt++;
        }
        c1.interchain_count_TD_BD_to_GR++;
        
    }
    else
    { // swap is up from GR to BD
        double likeT1X2     = c2.c_GR.loglike_loc*T2/T1; //likelihood of location 2 at temp 1
        double likeT2X1     = c1.c_BD.loglike_loc*T1/T2; //likelihood of location 1 at temp 2
        double rh           = min(1., exp((likeT1X2 + likeT2X1)-(c1.c_BD.loglike_loc + c2.c_GR.loglike_loc)));
        double urn          = gsl_ran_flat(r, 0, 1.);
        
        if(rh >= urn)
        {
            set_loc(c2.c_GR.loc, c1.c_GR.loc); // sets chain1's GR walker to that of chain2's
            set_loc(c1.c_BD.loc, c2.c_BD.loc); // sets chain2's BD walker to that of chain1's
            c1.c_GR.loglike_loc = likeT1X2;
            c2.c_BD.loglike_loc = likeT2X1;
            c1.gr_true = 1;
            c2.gr_true = 0;
            c1.interchain_count_TD_GR_to_BD_accpt++;
        }
        c1.interchain_count_TD_GR_to_BD++;
    }
    
}

void inter_chain_swap_same_dim(Chain_GR &c1, Chain_GR &c2, const gsl_rng * r)
{
    double T1       = c1.temp;
    double T2       = c2.temp;
    double likeT1X2 = c2.loglike_loc*T2/T1; //likelihood of location 2 at temp 1
    double likeT2X1 = c1.loglike_loc*T1/T2; //likelihood of location 1 at temp 2
    double rh       = min(1., exp((likeT1X2 + likeT2X1)-(c1.loglike_loc + c2.loglike_loc)));
    double urn      = gsl_ran_flat(r, 0, 1.);
    
    if(rh >= urn)
    {
        swap_loc(c1.loc, c2.loc);
        swap_fishers(c1.fisher, c2.fisher, c1.eigen_sys, c2.eigen_sys, T1, T2);
        c1.loglike_loc = likeT1X2;
        c2.loglike_loc = likeT2X1;
        c1.count_swap_accpt++;
    }
}
void inter_chain_swap_same_dim(Chain_BD &c1, Chain_BD &c2, const gsl_rng * r)
{
    double T1       = c1.temp;
    double T2       = c2.temp;
    double likeT1X2 = c2.loglike_loc*T2/T1; //likelihood of location 2 at temp 1
    double likeT2X1 = c1.loglike_loc*T1/T2; //likelihood of location 1 at temp 2
    double rh       = min(1., exp((likeT1X2 + likeT2X1) - (c1.loglike_loc + c2.loglike_loc)));
    double urn      = gsl_ran_flat(r, 0, 1.);
    
    if(rh >= urn)
    {
        swap_loc(c1.loc, c2.loc);
        swap_fishers(c1.fisher, c2.fisher, c1.eigen_sys, c2.eigen_sys, T1, T2);
        c1.loglike_loc = likeT1X2;
        c2.loglike_loc = likeT2X1;
        c1.count_swap_accpt++;
    }
}

void swap_fishers(Eigen::MatrixXd &fish1, Eigen::MatrixXd &fish2, Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> &es1 , Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> &es2, double T1, double T2)
{
    Eigen::MatrixXd fish_tmp;
    
    fish_tmp    = fish1;
    fish1       = fish2*T2/T1;
    fish2       = fish_tmp*T1/T2;
    
    es1.compute(fish1);
    es2.compute(fish2);
}

void swap_loc(vector<double> &loc1, vector<double> &loc2)
{
    vector<double> tmp(5);
    set_loc(loc1, tmp);
    set_loc(loc2, loc1);
    set_loc(tmp, loc2);
}

// Turn these into member functions

void record(Chain &c, vector<vector<vector<double>>> &chain_store, vector<vector<double>> &like_store, int chain_num, int i)
{
    if(c.gr_true == 1)
    {
        write_vec_to_vec(chain_store[chain_num], c.c_GR.loc, i);
        like_store[chain_num][i] = c.c_GR.loglike_loc;
    }
    else
    {
        write_vec_to_vec(chain_store[chain_num], c.c_BD.loc, i);
        like_store[chain_num][i] = c.c_BD.loglike_loc;
    }
}

void write_vec_to_vec(vector<vector<double>> &samples, vector<double> &sample, int i)
{
    samples[i][0] = sample[0];
    samples[i][1] = sample[1];
    samples[i][2] = sample[2];
    samples[i][3] = sample[3];
    samples[i][4] = sample[4];
}

void write_vec_to_file(vector<vector<double>> &vect, string filename, string path)
{
    ofstream out;
    out.open(path + filename);
    
    for (int i=0; i<vect.size(); i++)
    {
        out << setprecision(16)  << ' ' << vect[i][0] << ' ' << vect[i][1] << ' ' << vect[i][2] << ' ' << vect[i][3] << ' ' << vect[i][4] << endl;
    }
    out.close();
}

void write_vec_to_file(vector<vector<double>> &vect, string filename, string path, ofstream &out, int start, int end)
{
    if (start == 0){
        out.open(path + filename);
    } else {
        out.open(path + filename, ios::app);
    }
    for (int j = start; j < end; j++){
        out << setprecision(16)  << ' ' << vect[j][0] << ' ' << vect[j][1] << ' ' << vect[j][2] << ' ' << vect[j][3] << ' ' << vect[j][4] << endl;
    }
    out.close();
}

void write_vec_to_file(vector<double> &vect, string filename, string path, ofstream &out, int start, int end){
    if (start == 0){
        out.open(path + filename);
    } else {
        out.open(path + filename, ios::app);
    }
    for (int i = start; i < end; i++){
        out << setprecision(16)  << ' ' << vect[i] << endl;
    }
    out.close();
}

void write_vec_to_file(vector<double> &vect, string str, string path){
    ofstream out;
    out.open(path + str);
    for (int i=0; i<vect.size(); i++){
        out << setprecision(16) << ' ' << vect[i] << endl;
    }
    out.close();
}

void cout_chain_info(Chain c){
    cout << "GR CHAIN" << endl;
    cout << endl;
    cout_chain_info(c.c_GR);
    cout << "BD CHAIN" << endl;
    cout << endl;
    cout_chain_info(c.c_BD);
}

void cout_chain_info(Chain_GR c){
    cout << "Temp = " << c.temp << endl;
    cout << "like_loc = " << c.loglike_loc << endl;
    cout << "like_prop = " << c.loglike_prop << endl;
    cout << "hast = " << c.hast_ratio << endl;
    cout << "urv = " << c.urv << endl;
    cout << "count temp = " << c.count_in_temp << endl;
    cout << "count temp = " << c.count_swap << endl;
    cout << "location" << endl;
    cout_vec(c.loc);
    cout << "proposal" << endl;
    cout_vec(c.prop);
    cout << "Fisher" << endl;
    cout << c.fisher << endl;
    cout << "Eigensys" << endl;
    cout << c.eigen_sys.eigenvalues() << endl;
    cout << c.eigen_sys.eigenvectors() << endl;
}

void cout_chain_info(Chain_BD c){
    cout << "Temp = " << c.temp << endl;
    cout << "like_loc = " << c.loglike_loc << endl;
    cout << "like_prop = " << c.loglike_prop << endl;
    cout << "hast = " << c.hast_ratio << endl;
    cout << "urv = " << c.urv << endl;
    cout << "count temp = " << c.count_in_temp << endl;
    cout << "count temp = " << c.count_swap << endl;
    cout << "location" << endl;
    cout_vec(c.loc);
    cout << "proposal" << endl;
    cout_vec(c.prop);
    cout << "Fisher" << endl;
    cout << c.fisher << endl;
    cout << "Eigensys" << endl;
    cout << c.eigen_sys.eigenvalues() << endl;
    cout << c.eigen_sys.eigenvectors() << endl;
}

void cout_vec(vector<double> &vec){
    cout << " M = " << vec[0] <<  " eta = " << vec[1] <<  " e0 = " << vec[2] <<  " amp = " << vec[3] << " b = " << vec[4]  << endl;
}

void print_accpt_ratios(vector<Chain> &c_vect){
    int N = c_vect.size();
    for(int i = 0; i < N; i++){
        print_accpt_ratios(c_vect[i], i);
    }
}

void print_accpt_ratios(Chain &c, int i){
    cout << "ACCEPTANCE RATIOS FOR CHAIN # " << i << endl << endl;
    cout << "   Within Dimension: " << endl << endl;
    cout << "       GR:" << endl;
    cout << "           Within Temp:" <<  (double) c.c_GR.count_in_temp_accpt/c.c_GR.count_in_temp << endl;
    cout << "           Interchain:" <<  (double) c.c_GR.count_swap_accpt/c.c_GR.count_swap << endl;
    cout << endl;
    cout << "       BD:" << endl;
    cout << "           Within Temp:" <<  (double) c.c_BD.count_in_temp_accpt/c.c_BD.count_in_temp << endl;
    cout << "           Interchain:" <<  (double) c.c_BD.count_swap_accpt/c.c_BD.count_swap << endl;
    cout << endl;
    cout << "   Transdimensional:" << endl << endl;
    cout << "       GR -> BD" << endl;
    cout << "           Within Temp:" << (double) c.count_TD_GR_to_BD_accpt/c.count_TD_GR_to_BD << endl;
    cout << "           Interchain:"  << (double) c.interchain_count_TD_GR_to_BD_accpt/c.interchain_count_TD_GR_to_BD << endl;
    cout << endl;
    cout << "       BD -> GR" << endl;
    cout << "           Within Temp:" << (double) c.count_TD_BD_to_GR_accpt/c.count_TD_BD_to_GR << endl;
    cout << "           Interchain:"  << (double) c.interchain_count_TD_BD_to_GR_accpt/c.interchain_count_TD_BD_to_GR << endl;
    cout << endl;
}
