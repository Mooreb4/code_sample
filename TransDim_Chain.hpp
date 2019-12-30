#ifndef _TRANSDIM_CHAIN_HPP_
#define _TRANSDIM_CHAIN_HPP_

#include "Chain.hpp"

//Interface Class
class TransDim_Chain
{
public:
    virtual ~TransDim_Chain() {};
    virtual void transdim_jump()                             = 0;
    virtual void update_prop_1_to_2()                        = 0;
    virtual void update_prop_2_to_1()                        = 0;
    virtual void transdim_jump_2_to_1()                      = 0;
    virtual void accept_2_to_1()                             = 0;
    virtual void reject_2_to_1()                             = 0;
    virtual void transdim_interchain_swap(TransDim_Chain &c) = 0;
    virtual void jump()                                      = 0;
    virtual void transdim_jump_1_to_2()                      = 0;
};

class TransDim_Chain_BD_GR: public TransDim_Chain
{
public:
    TransDim_Chain_BD_GR(vector<complex<double>> &signal_, vector<double> curr_BD, vector<double> curr_GR, vector<double> &noise_, vector<double> &noise_fish_, double temp_, double f_begin_, double fend_, double df_, double df_fish_, double ep_fish_, unsigned int num_params_BD_,  unsigned int num_params_GR_, unsigned int num_diff_evol_samples_, bool init_state);
    virtual ~TransDim_Chain_BD_GR();
    virtual void update_prop_1_to_2();
    virtual void update_prop_2_to_1();
    virtual void transdim_jump_2_to_1();
    virtual void accept_2_to_1();
    virtual void reject_2_to_1();
    virtual void transdim_jump();
    virtual void transdim_interchain_swap();
    virtual void jump();
    virtual void transdim_jump_1_to_2();
    
private:
    Chain_GR c_GR;
    Chain_BD c_BD;
    unsigned int num_params_GR;
    unsigned int num_params_BD;
    unsigned int count_transdim;
    unsigned int count_transdim_accpt;
    unsigned int count_transdim_interchain;
    unsigned int count_transdim_interchain_accpt;
    bool is_curr_GR;
    gsl_rng * r;
};
#endif
