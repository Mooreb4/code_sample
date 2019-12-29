#include "Chain.hpp"

// Implement Chain_GR

// Default Constructor
Chain_GR::Chain_GR(vector<complex<double>> signal_, vector<double> curr, vector<double> noise_, vector<double> noise_fish_, double temp_, double f_begin_, double fend_, double df_, double df_fish_, double ep_fish_, unsigned int num_params_, unsigned int num_diff_evol_samples_)
{
    count_in_temp           = 0;
    count_interchain        = 0;
    count_in_temp_accpt     = 0;
    count_interchain_accpt  = 0;
    diff_evol_track         = 0;
    curr_state              = curr;
    temp                    = temp_;
    curr_log_like           = loglike(curr_state, f_begin, fend, df, signal, noise, temp);
    fisher                  = fim_GR(curr_state, noise_fish, f_begin, fend, df_fish, ep_fish, temp, 3);
    out_of_prior_bounds     = false;
    
    signal                  = signal_;
    noise                   = noise_;
    noise_fish              = noise_fish_;
    f_begin                 = f_begin_;
    fend                    = fend_;
    df                      = df_;
    df_fish                 = df_fish_;
    ep_fish                 = ep_fish_;
    num_params              = num_params_;
    num_diff_evol_samples   = num_diff_evol_samples_;
    
    prop_state.resize(num_params);
    diff_evol_vals.resize(num_diff_evol_samples, vector<double> (num_params));
    eigen_sys.compute(fisher);
    r = gsl_rng_alloc (gsl_rng_taus);
}
// Destructor
Chain_GR::~Chain_GR()
{
    gsl_rng_free (r);
}
// copy constructor
Chain_GR::Chain_GR( const Chain_GR &copy)
{
    count_in_temp           = copy.count_in_temp;
    count_interchain        = copy.count_interchain;
    count_in_temp_accpt     = copy.count_in_temp_accpt;
    count_interchain_accpt  = copy.count_interchain_accpt;
    diff_evol_track         = copy.diff_evol_track;
    curr_state              = copy.curr_state;
    temp                    = copy.temp;
    curr_log_like           = copy.curr_log_like;
    fisher                  = copy.fisher;
    out_of_prior_bounds     = copy.out_of_prior_bounds;
    
    signal                  = copy.signal;
    noise                   = copy.noise;
    noise_fish              = copy.noise_fish;
    f_begin                 = copy.f_begin;
    fend                    = copy.fend;
    df                      = copy.df;
    df_fish                 = copy.df_fish;
    ep_fish                 = copy.ep_fish;
    num_params              = copy.num_params;
    num_diff_evol_samples   = copy.num_diff_evol_samples;
    
    prop_state              = copy.prop_state;
    diff_evol_vals          = copy.diff_evol_vals;
    eigen_sys               = copy.eigen_sys;
    
    r                       = gsl_rng_clone( copy.r );
}

// copy assigment constructor
Chain_GR& Chain_GR::operator=( const Chain_GR &rhs )
{
    if ( this != &rhs)
    {
        count_in_temp           = rhs.count_in_temp;
        count_interchain        = rhs.count_interchain;
        count_in_temp_accpt     = rhs.count_in_temp_accpt;
        count_interchain_accpt  = rhs.count_interchain_accpt;
        diff_evol_track         = rhs.diff_evol_track;
        curr_state              = rhs.curr_state;
        temp                    = rhs.temp;
        curr_log_like           = rhs.curr_log_like;
        fisher                  = rhs.fisher;
        out_of_prior_bounds     = rhs.out_of_prior_bounds;
        
        signal                  = rhs.signal;
        noise                   = rhs.noise;
        noise_fish              = rhs.noise_fish;
        f_begin                 = rhs.f_begin;
        fend                    = rhs.fend;
        df                      = rhs.df;
        df_fish                 = rhs.df_fish;
        ep_fish                 = rhs.ep_fish;
        num_params              = rhs.num_params;
        num_diff_evol_samples   = rhs.num_diff_evol_samples;
        
        prop_state              = rhs.prop_state;
        diff_evol_vals          = rhs.diff_evol_vals;
        eigen_sys               = rhs.eigen_sys;
        
        r                       = gsl_rng_clone( rhs.r );
    }
}

void Chain_GR::update_prop_fisher()
{
    double delt = gsl_ran_gaussian (r, 1.);
    int i = floor(gsl_ran_flat(r, 0, 4));
    if (i == 4){i = 3;}
    prop_state[0] = curr_state[0]*exp(delt*1./sqrt((eigen_sys.eigenvalues()(i)))*eigen_sys.eigenvectors().col(i)(0));
    prop_state[1] = curr_state[1] + delt*1./sqrt((eigen_sys.eigenvalues()(i)))*eigen_sys.eigenvectors().col(i)(1);
    prop_state[2] = curr_state[2] + delt*1./sqrt((eigen_sys.eigenvalues()(i)))*eigen_sys.eigenvectors().col(i)(2);
    prop_state[3] = curr_state[3]*exp(delt*1./sqrt((eigen_sys.eigenvalues()(i)))*eigen_sys.eigenvectors().col(i)(3));
}

void Chain_GR::update_prop_diff_evol()
{
    if(diff_evol_track < 2) //Not enough diff evol samples to do a diff evol jump
    {
        update_prop_fisher();
    }
    else
    {
        double fact = gsl_ran_gaussian (r, 0.751319);
        int i       = floor(gsl_ran_flat(r, 0, diff_evol_track - 1));
        int j       = floor(gsl_ran_flat(r, 0, diff_evol_track - 1));
        prop_state[0]     = exp(log(curr_state[0]) + fact*(log(diff_evol_vals[i][0]) - log(diff_evol_vals[j][0])));
        prop_state[1]     = curr_state[1] + fact*(diff_evol_vals[i][1] - diff_evol_vals[j][1]);
        prop_state[2]     = curr_state[2] + fact*(diff_evol_vals[i][2] - diff_evol_vals[j][2]);
        prop_state[3]     = exp(log(curr_state[3]) + fact*(log(diff_evol_vals[i][3]) - log(diff_evol_vals[j][3])));
    }
}

//is there a better way to handle priors without hard coding them here? own class?
void Chain_GR::update_prop_priors()
{
    prop_state[0] = exp(gsl_ran_flat(r, -0.223144, 2.30259));
    prop_state[1] = gsl_ran_flat(r, 0.08, 0.25);
    prop_state[2] = gsl_ran_flat(r, 0.000001, 0.805);
    prop_state[3] = exp(gsl_ran_flat(r, -46.0517, -36.8414));
}

void Chain_GR::calc_log_like_prop()
{
    prop_log_like = loglike(prop_state, f_begin, fend, df, signal, noise, temp);
}

void Chain_GR::check_prior()
{
    out_of_prior_bounds = false;
    
    if(prop_state[0] > 10     || prop_state[0] < 0.8)       {out_of_prior_bounds = true;}
    if(prop_state[1] > 0.25   || prop_state[1] < 0.08)      {out_of_prior_bounds = true;}
    if(prop_state[2] > 0.805  || prop_state[2] < 0.000001)  {out_of_prior_bounds = true;}
    if(prop_state[3] > 1e-16  || prop_state[3] < 1e-20)     {out_of_prior_bounds = true;}
}

void Chain_GR::attempt_jump()
{
    if(out_of_prior_bounds == true)
    {
        reject_jump();
    }
    else
    {
        double hastings_ratio = min(1., prop_log_like - curr_log_like);
        double uniform_RV     = gsl_ran_flat(r, 0, 1.);
        
        if(hastings_ratio >= uniform_RV)
        {
            accept_jump();
        }
        else
        {
            reject_jump();
        }
    }
}

void Chain_GR::accept_jump()
{
    count_in_temp++;
    count_in_temp_accpt++;
    
    curr_log_like = prop_log_like;
    
    for(int i = 0; i < curr_state.size(); i++)
    {
        curr_state[i] = prop_state[i];
    }
}

void Chain_GR::reject_jump()
{
    count_in_temp++;
}

void Chain_GR::jump()
{
    double uniform_RV = gsl_ran_flat(r, 0, 1.);
    
    if(uniform_RV < 0.05)
    {
        update_prop_priors();
    }
    else if(uniform_RV < 0.30)
    {
        update_prop_diff_evol();
    }
    else
    {
        update_prop_fisher();
    }
    
    check_prior();
    attempt_jump();
}

void Chain_GR::print_acc_ratios()
{
    cout << "GR chain with temp "   << temp << " acceptance ratios:"                    << endl;
    cout << "Within Temp: "         << (double) count_in_temp_accpt/count_in_temp       << endl;
    cout << "Interchain: "          << (double) count_interchain_accpt/count_interchain << endl;
    cout << endl;
}

void Chain_GR::print_states()
{
    cout << "The current state:" << endl;
    for(auto x : curr_state)
    {
        cout << x << endl;
    }
    cout << endl;
    
    cout << "The proposed state:" << endl;
    for(auto x : prop_state)
    {
        cout << x << endl;
    }
    cout << endl;
}

void Chain_GR::print_fisher()
{
    cout << "The fisher information matrix:" << endl;
    cout << fisher << endl << endl;
    cout << "The eigenvalues of the fisher:" << endl;
    cout << eigen_sys.eigenvalues().transpose() << endl << endl;
    cout << "The eigenvectors of the fisher:" << endl;
    cout << eigen_sys.eigenvectors() << endl << endl;
}

void Chain_GR::print_all()
{
    print_acc_ratios();
    print_states();
    print_fisher();
}

void Chain_GR::write_to_diff_evol()
{
    int ind = diff_evol_track % num_diff_evol_samples;
    
    for(int i = 0; i < num_params; i++)
    {
        diff_evol_vals[ind][i] = curr_state[i];
    }
}

void Chain_GR::update_fisher()
{
    fisher = fim_GR(curr_state, noise_fish, f_begin, fend, df_fish, ep_fish, temp, 3);
}
