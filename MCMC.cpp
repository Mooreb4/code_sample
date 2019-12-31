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
#include "Chain.hpp"

unsigned int num_diff_evol_samples = 1000;
auto f_begin = 0;
auto f_end = 1000;
auto df = 0.015625;
auto ep_fish = 1e-8;
auto df_fish = 0.25;

vector<complex<double>> gen_waveform(double M, double eta, double e0, double A, double b, double fend);
vector<complex<double>> sum_f2(vector<vector<complex<double>>> &vect);

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
    
    auto h2  = gen_waveform(M_in, eta_in, e0_in, A_in, b_in, f_end);
    auto data_size               = h2.size();
    
    vector<double> freqs(data_size);
    
    for(unsigned int i = 0; i < freqs.size(); i++)
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
    for (unsigned int i = 0; i < noise.size(); i++)
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
    
    auto N_down_noise = std::floor(fend/df_fish) + 1;
    
    vector<double> noise_fish(N_down_noise);
    auto f = 0.0;
    for (unsigned int i = 0; i < N_down_noise; i++)
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
    
    auto A_prev = A_in;
    auto A_curr = A_in - A_in/100;
    
    auto h1 = gen_waveform(M_in, eta_in, e0_in, A_prev, b_in, fend);
    h2 = gen_waveform(M_in, eta_in, e0_in, A_curr, b_in, fend);
    
    auto snr_shoot_1      = sqrt(get_snr_sq(h1, noise, df));
    auto snr_shoot_2      = sqrt(get_snr_sq(h2, noise, df));
    auto snr_diff_prev    = snr_shoot_1 - SNR_in;
    auto snr_diff_curr    = snr_shoot_2 - SNR_in;
    auto A_next           = A_prev - snr_diff_prev*(A_prev-A_curr)/(snr_diff_prev-snr_diff_curr);

    while(abs(snr_diff_curr) > 0.001)
    {
        A_prev              = A_curr;
        snr_diff_prev       = snr_diff_curr;
        A_curr              = A_next;
        h2                  = gen_waveform(M_in, eta_in, e0_in, A_curr, b_in, fend);
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
    
    auto N_chain          = stoi(argv[7]);
    auto temp_spacing     = stod(argv[8]);
    auto chain_type       = stoi(argv[10]);
    
    vector<Chain*> chains;
    vector<double> GR_loc(5);
    vector<double> BD_loc(5);
    
    GR_loc[0] = M_in;
    GR_loc[1] = eta_in;
    GR_loc[2] = e0_in;
    GR_loc[3] = A_curr;
    GR_loc[4] = 0;
    
    BD_loc[0] = M_in;
    BD_loc[1] = eta_in;
    BD_loc[2] = e0_in;
    BD_loc[3] = A_curr;
    BD_loc[4] = b_in;
    
    //initialize many chains
    if(chain_type == 1)
    {
        for(unsigned int i = 0; i < N_chain; i++)
        {
            Chain * c = new Chain_GR(h2, GR_loc, noise, noise_fish, pow(temp_spacing, i), f_begin, fend, df, df_fish, ep_fish, 5, num_diff_evol_samples);
            chains.push_back(c);
        }
    }
    else
    {
        for(unsigned int i = 0; i < N_chain; i++)
        {
            Chain * c = new Chain_BD(h2, BD_loc, noise, noise_fish, pow(temp_spacing, i), f_begin, fend, df, df_fish, ep_fish, 5, num_diff_evol_samples);
            chains.push_back(c);
        }
    }
    
    std::cout << "Chains inititialized" << std::endl;
    ////////////////////////////////////////////////////////
    // MCMC Routine
    ////////////////////////////////////////////////////////
    
    ofstream output;
    
    auto N_jumps        = stoi(argv[9]);
    
    for(unsigned int i = 0; i < N_jumps; i++)
    {
        if( i % 5 != 0)
        { //Within Tempurature jumps
            for(auto c : chains)
            {
                c -> jump();
            }
        }
        else
        { //Inter tempurature jumps
            for(unsigned int j = 0; j < N_chain - 1; j++)
            {
                chains[j] -> interchain_swap(*chains[j + 1]);
            }
        }
        //Write to differential evolution list every 100 jumps
        if ( i % 100 == 0)
        {
            for(auto c : chains)
            {
                c -> write_to_diff_evol();
            }
        }
        //Print info periodically
        if (i % 10000 == 0)
        {
            for(auto c : chains)
            {
                c -> print_all();
            }
        }
        //Periodically update Fishers.
        if( i % 800 == 0 )
        {
            for(auto c : chains)
            {
                c -> update_fisher();
            }
        }
    }
    
    for(auto c : chains)
    {
        delete c;
    }
    
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
