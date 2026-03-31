#ifndef _MOETRAFFICMANAGER_ACCELSIM_HPP_
#define _MOETRAFFICMANAGER_ACCELSIM_HPP_

#include "batchtrafficmanager.hpp"
#include <vector>
#include <string>

// Traffic manager for hbmnet_accelsim topology.
//
// Router ID layout:
//   Router 0,1       : Crossbar partitions (Xbar0, Xbar1)
//   Router 2..K+1    : MC routers 0..K-1
//   Router K+2..2K+1 : HBM routers 0..K-1
//
// Traffic matrix file format (tab-separated):
//   Same as MoETrafficManager — supports Xbar, HBM, SM, L2 labels.
//   MC routers are internal (no traffic matrix labels needed).

class MoETrafficManagerAccelSim : public BatchTrafficManager {

protected:

  vector<vector<int>> _moe_matrix;
  vector<vector<int>> _moe_remaining;
  vector<int> _moe_dest_rr;

  int _moe_total_packets;
  int _moe_received_packets;

  double _moe_total_mb;
  int _flit_width_bytes;

  double _moe_injection_rate;

  void _LoadTrafficMatrix(const Configuration &config);

  virtual void _Inject() override;
  virtual bool _SingleSim() override;

public:

  MoETrafficManagerAccelSim(const Configuration &config, const vector<Network *> &net);
  virtual ~MoETrafficManagerAccelSim();

};

#endif
