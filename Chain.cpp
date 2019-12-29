#include <Chain.hpp>

// Implement Chain_GR

// Default Constructor
Chain_GR::Chain_GR()
{
    r = gsl_rng_alloc (gsl_rng_taus);
}
// Destructor
Chain_GR::~Chain_GR()
{
    gsl_rng_free(r);
}
// todo BCM -- make copy constructor and copy assignment constructor

// Going to let an init function handle initializing values as it requires some inputs
// MOVE THIS INIT TO THE CONSTRUCTOR
void Chain_GR::init_walker(vector<complex<double>> signal, vector<double> curr, vector<double> noise, double T, double f_begin, double fend, double df, double df_fish, double ep_fish, int num_params, int num_diff_evol_samples)
{
    count_in_temp           = 0;
    count_intechain         = 0;
    count_in_temp_accpt     = 0;
    count_interchain_appct  = 0;
    diff_evol_track         = 0;
    curr_state              = curr;
    temp                    = T;
    curr_log_like           = loglike(curr_state, f_begin, fend, df, signal, noise, T);
    fisher                  = fim_GR(curr_state, noise_fish, f_begin, fend, df_fish, ep_fish, T, 3);
    out_of_prior_bounds     = false;
    
    prop_state.resize(num_params);
    diff_evol_vals.resize(num_diff_evol_samples, vector<double> (num_params));
    eigen_sys.compute(fisher);
}

// Consider just storing the log parameter in the below so I can just for the assignments
void Chain_GR::update_prop_fisher()
{
    double delt = gsl_ran_gaussian (r, 1.);
    int i = floor(gsl_ran_flat(r, 0, 4));
    if (i == 4){i = 3;}
    prop_state[0] = curr_state[0]*exp(delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(0));
    prop_state[1] = curr_state[1] + delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(1);
    prop_state[2] = curr_state[2] + delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(2);
    prop_state[3] = curr_state[3]*exp(delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(3));
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
        prop[0]     = exp(log(curr_state[0]) + fact*(log(diff_evol_vals[i][0]) - log(diff_evol_vals[j][0])));
        prop[1]     = curr_state[1] + fact*(diff_evol_vals[i][1] - diff_evol_vals[j][1]);
        prop[2]     = curr_state[2] + fact*(diff_evol_vals[i][2] - diff_evol_vals[j][2]);
        prop[3]     = exp(log(curr_state[3]) + fact*(log(diff_evol_vals[i][3]) - log(diff_evol_vals[j][3])));
    }
}

//is there a better way to handle priors without hard coding them here? own class?
void Chain_GR::update_prop_priors()
{
    prop[0] = exp(gsl_ran_flat(r, -0.223144, 2.30259));
    prop[1] = gsl_ran_flat(r, 0.08, 0.25);
    prop[2] = gsl_ran_flat(r, 0.000001, 0.805);
    prop[3] = exp(gsl_ran_flat(r, -46.0517, -36.8414));
}

void Chain_GR::calc_log_like_prop()
{
    prop_log_like = loglike(prop_state, f_begin, fend, df, signal, noise, temp);
}

void Chain_GR::check_prior()
{
    out_of_prior_bounds = false;
    
    if(prop[0] > 10     || prop[0] < 0.8)       {out_of_prior_bounds = true;}
    if(prop[1] > 0.25   || prop[1] < 0.08)      {out_of_prior_bounds = true;}
    if(prop[2] > 0.805  || prop[2] < 0.000001)  {out_of_prior_bounds = true;}
    if(prop[3] > 1e-16  || prop[3] < 1e-20)     {out_of_prior_bounds = true;}
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
    cout << "GR chain with temp "   << temp << " acceptance ratios:"                   << endl;
    cout << "Within Temp: "         << (double) count_in_temp_accpt/count_in_temp       << endl;
    cout << "Interchain: "     << (double) count_interchain_appct/count_interchain << endl;
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
    cout << es.eigenvalues().transpose() << endl << endl;
    cout << "The eigenvectors of the fisher:" << endl;
    cout << es.eigenvectors() << endl << endl;
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
