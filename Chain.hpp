#ifndef _CHAIN_HPP_
#define _CHAIN_HPP_

#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#include "llhood_maxd.hpp"
#include "fisher.hpp"

// Interface Class
class Chain
{
public:
    virtual ~Chain() {};
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
};

// A derived class which handles General Relativity Parameter Estimation
class Chain_GR : public Chain
{
public:
    Chain_GR(vector<complex<double>> signal_, vector<double> curr, vector<double> noise_, vector<double> noise_fish_, double temp_, double f_begin_, double fend_, double df_, double df_fish_, double ep_fish_, unsigned int num_params_, unsigned int num_diff_evol_samples_);
    virtual ~Chain_GR();
    Chain_GR( const Chain_GR &copy);
    Chain_GR& operator=( const Chain_GR &rhs);
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
    unsigned int num_params;
    unsigned int num_diff_evol_samples;
    double curr_log_like;
    double prop_log_like;
    double temp;
    double f_begin;
    double fend;
    double df;
    double df_fish;
    double ep_fish;
    gsl_rng * r;
    vector<double> curr_state;
    vector<double> prop_state;
    vector<double> noise;
    vector<double> noise_fish;
    vector<vector<double>> diff_evol_vals;
    vector<complex<double>> signal;
    Eigen::MatrixXd fisher;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_sys;
};

// A derived class which handles Brans-Dicke gravity Parameter Estimation
class Chain_BD : public Chain
{
public:
    Chain_BD(vector<complex<double>> signal_, vector<double> curr, vector<double> noise_, vector<double> noise_fish_, double temp_, double f_begin_, double fend_, double df_, double df_fish_, double ep_fish_, unsigned int num_params_, unsigned int num_diff_evol_samples_);
    virtual ~Chain_BD();
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
    unsigned int num_params;
    unsigned int num_diff_evol_samples;
    double curr_log_like;
    double prop_log_like;
    double temp;
    double f_begin;
    double fend;
    double df;
    double df_fish;
    double ep_fish;
    gsl_rng * r;
    vector<double> curr_state;
    vector<double> prop_state;
    vector<double> noise;
    vector<double> noise_fish;
    vector<vector<double>> diff_evol_vals;
    vector<complex<double>> signal;
    Eigen::MatrixXd fisher;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_sys;
};

#endif
