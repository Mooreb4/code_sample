#include "TransDim_Chain.hpp"


////////////////////////////////////////////////////////////////////////////////
//
// Implementing TransDim_Chain
//
////////////////////////////////////////////////////////////////////////////////

// Constructor

TransDim_Chain_BD_GR::TransDim_Chain_BD_GR(vector<complex<double>> &signal_, vector<double> curr_BD, vector<double> curr_GR, vector<double> &noise_, vector<double> &noise_fish_, double temp_, double f_begin_, double fend_, double df_, double df_fish_, double ep_fish_, unsigned int num_params_BD_,  unsigned int num_params_GR_, unsigned int num_diff_evol_samples_, bool init_state):

    c_BD(signal_, curr_BD, noise_, noise_fish_, temp_, f_begin_, fend_, df_, df_fish_, ep_fish_, num_params_BD_, num_diff_evol_samples_),
    c_GR(signal_, curr_GR, noise_, noise_fish_, temp_, f_begin_, fend_, df_, df_fish_, ep_fish_, num_params_GR_, num_diff_evol_samples_)
{
    num_params_GR = num_params_GR_;
    num_params_BD = num_params_BD_;
    count_transdim_GR_BD = 0;
    count_transdim_GR_BD_accpt = 0;
    count_transdim_GR_BD_interchain = 0;
    count_transdim_GR_BD_interchain_accpt = 0;
    count_transdim_BD_GR = 0;
    count_transdim_BD_GR_accpt = 0;
    count_transdim_BD_GR_interchain = 0;
    count_transdim_BD_GR_interchain_accpt = 0;
    is_curr_GR = init_state;
    r = gsl_rng_alloc (gsl_rng_taus);
}
// Destructor
TransDim_Chain_BD_GR::~TransDim_Chain_BD_GR()
{
    gsl_rng_free (r);
}
//TODO -- BCM -- Make a copy and copy assignement constructor
void TransDim_Chain_BD_GR::update_prop_1_to_2() //BD to GR
{
    for(int i = 0; i < c_GR.prop_state.size(); i++)
    {
        c_GR.prop_state[i] = c_BD.curr_state[i];
    }
}

void TransDim_Chain_BD_GR::update_prop_2_to_1() //GR to BD
{
    for(int i = 0 ; i < c_GR.curr_state.size(); i++)
    {
        c_BD.prop_state[i] = c_GR.curr_state[i];
    }
    c_BD.prop_state[4] = exp(gsl_ran_flat(r, -23.0259, -4.60517));
}

void TransDim_Chain_BD_GR::transdim_jump_2_to_1()
{
    update_prop_2_to_1();
    c_BD.check_prior();
    if(c_BD.out_of_prior_bounds == true)
    {
        reject_2_to_1();
    }
    else
    {
        double p_x = log(1./c_BD.prop_state[4]); //the extra-dimensional RV draw PDF evaluated at extra-dimensional value
        c_BD.calc_log_like_prop();
        double transdim_log_like_prop = c_BD.prop_log_like;
        double uniform_RV = gsl_ran_flat(r, 0, 1.);
        double hastings_ratio = min(1., 2825.16*exp(transdim_log_like_prop - c_GR.curr_log_like - p_x));  //hard coded is contribution from priors
        if(hastings_ratio >= uniform_RV)
        {
            accept_2_to_1();
        }
        else
        {
            reject_2_to_1();
        }
    }
}

void TransDim_Chain_BD_GR::transdim_jump_1_to_2()
{
    update_prop_1_to_2();
    c_GR.check_prior();
    if(c_GR.out_of_prior_bounds == true)
    {
        reject_1_to_2();
    }
    else
    {
        double p_x = log(1./c_BD.curr_state[4]*0.0542868); //the extra-dimensional RV draw PDF evaluated at extra-dimensional value
        c_GR.calc_log_like_prop();
        double transdim_log_like_prop = c_GR.prop_log_like;
        double uniform_RV = gsl_ran_flat(r, 0, 1.);
        double hastings_ratio = min(1., 0.0003539626*exp(transdim_log_like_prop - c_BD.curr_log_like + p_x));  //hard coded is contribution from priors
        if(hastings_ratio >= uniform_RV)
        {
            accept_1_to_2();
        }
        else
        {
            reject_1_to_2();
        }
    }
}

void TransDim_Chain_BD_GR::accept_2_to_1()
{
    c_BD.accept_jump();
    c_BD.count_in_temp--;
    c_BD.count_in_temp_accpt--;
    count_transdim_GR_BD++;
    count_transdim_GR_BD_accpt++;
}

void TransDim_Chain_BD_GR::accept_1_to_2()
{
    c_GR.accept_jump();
    c_GR.count_in_temp--;
    c_GR.count_in_temp_accpt--;
    count_transdim_BD_GR++;
    count_transdim_BD_GR_accpt++;
}

void TransDim_Chain_BD_GR::reject_2_to_1()
{
    count_transdim_GR_BD++;
}

void TransDim_Chain_BD_GR::reject_1_to_2()
{
    count_transdim_BD_GR++;
}

void TransDim_Chain_BD_GR::transdim_jump()
{
    if (is_curr_GR = true)
    {
        transdim_jump_2_to_1();
    }
    else
    {
        transdim_jump_1_to_2();
    }
}
