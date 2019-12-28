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
    virtual void jump(int i)                = 0;
    virtual void print_acc_ratios()         = 0;
    virtual void print_states()             = 0;
    virtual void print_fisher()             = 0;
    virtual void print_all()                = 0;
    virtual void write_to_diff_evol()       = 0;
    virtual void update_fisher()            = 0;
    virtual void accept_jump()              = 0;
    virtual void reject_jump()              = 0;
}

// An instance of a chain which is restricted to General Relativity waveforms
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
    virtual void jump(int i);
    virtual void print_acc_ratios();
    virtual void print_states();
    virtual void print_fisher();
    virtual void print_all();
    virtual void write_to_diff_evol();
    virtual void update_fisher();
    virtual void accept_jump();
    virtual void reject_jump();
    
private:
    int count_in_temp, count_interchain, count_transdim, count_in_temp_accpt,
        count_interchain_accpt, count_transdim_accpt, diff_evol_track;
    double curr_log_like, prop_log_like, temp;
    vector<double> curr_state, prop_state;
    vector<vector<double>> diff_evol_vals;
    Eigen::MatrixXd fisher;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_sys;
}

// An instance of a chain which can explore waveforms in Brans-Dicke Gravity
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
    virtual void jump(int i);
    virtual void print_acc_ratios();
    virtual void print_states();
    virtual void print_fisher();
    virtual void print_all();
    virtual void write_to_diff_evol();
    virtual void update_fisher();
    virtual void accept_jump();
    virtual void reject_jump();
    
private:
    int count_in_temp, count_interchain, count_transdim, count_in_temp_accpt,
    count_interchain_accpt, count_transdim_accpt, diff_evol_track;
    double curr_log_like, prop_log_like, temp;
    vector<double> curr_state, prop_state;
    vector<vector<double>> diff_evol_vals;
    Eigen::MatrixXd fisher;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_sys;
}
