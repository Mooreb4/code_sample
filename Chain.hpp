#ifndef _CHAIN_HPP_
#define _CHAIN_HPP_

#include <gsl/gsl_randist.h>

#include "llhood_maxd.hpp"
#include "fisher.hpp"

// Interface Class
class Chain
{
public:
    virtual ~Chain() {};
    virtual void init_walker()              = 0;
    virtual void update_prop_fisher()       = 0;
    virtual void update_prop_diff_evol()    = 0;
    virtual void update_prop_priors()       = 0;
    virtual void calc_log_like_prop()       = 0;
    virtual void check_prior()              = 0;
    virtual void attempt_jump()             = 0;
    virtual void jump()                     = 0;
    virtual void print_acc_ratios()         = 0;
    virtual void print_states()             = 0;
    virtual void print_fisher()             = 0;
    virtual void print_all()                = 0;
    virtual void write_to_diff_evol()       = 0;
    virtual void update_fisher()            = 0;
    virtual void accept_jump()              = 0;
    virtual void reject_jump()              = 0;
    virtual void interchain_swap(Chain c)   = 0;
}

// A derived class which handles General Relativity Parameter Estimation
class Chain_GR : public Chain
{
public:
    Chain_GR();
    virtual ~Chain_GR();
    virtual void init_walker();
    virtual void update_prop_fisher();
    virtual void update_prop_diff_evol();
    virtual void update_prop_priors();
    virtual void calc_log_like_prop();
    virtual void check_prior();
    virtual void attempt_jump();
    virtual void jump();
    virtual void print_acc_ratios();
    virtual void print_states();
    virtual void print_fisher();
    virtual void print_all();
    virtual void write_to_diff_evol();
    virtual void update_fisher();
    virtual void accept_jump();
    virtual void reject_jump();
    virtual void interchain_swap(Chain_GR &c);
    
private:
    bool out_of_prior_bounds;
    unsigned int count_in_temp;
    unsigned int count_interchain;
    unsigned int count_in_temp_accpt;
    unsigned int count_interchain_accpt;
    unsigned int diff_evol_track;
    double curr_log_like;
    double prop_log_like;
    double temp;
    const gsl_rng * r;
    vector<double> curr_state;
    vector<double> prop_state;
    vector<vector<double>> diff_evol_vals;
    Eigen::MatrixXd fisher;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_sys;
}

// A derived class which handles Brans-Dicke gravity Parameter Estimation
class Chain_BD : public Chain
{
public:
    Chain_BD();
    virtual ~Chain_BD();
    virtual void init_walker();
    virtual void update_prop_fisher();
    virtual void update_prop_diff_evol();
    virtual void update_prop_priors();
    virtual void calc_log_like_prop();
    virtual void check_prior();
    virtual void attempt_jump();
    virtual void jump();
    virtual void print_acc_ratios();
    virtual void print_states();
    virtual void print_fisher();
    virtual void print_all();
    virtual void write_to_diff_evol();
    virtual void update_fisher();
    virtual void accept_jump();
    virtual void reject_jump();
    virtual void interchain_swap(Chain_BD &c);
    
private:
    bool out_of_prior_bounds;
    unsigned int count_in_temp;
    unsigned int count_interchain;
    unsigned int count_in_temp_accpt;
    unsigned int count_interchain_accpt;
    unsigned int diff_evol_track;
    double curr_log_like;
    double prop_log_like;
    double temp;
    const gsl_rng * r;
    vector<double> curr_state;
    vector<double> prop_state;
    vector<vector<double>> diff_evol_vals;
    Eigen::MatrixXd fisher;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_sys;
}

#endif
