#ifndef _MOETRAFFICMANAGER_HPP_
#define _MOETRAFFICMANAGER_HPP_

#include "batchtrafficmanager.hpp"
#include <vector>
#include <string>

// Traffic manager for hbmnet topology.
//
// Router ID layout:
//   Router 0         : Crossbar partition 0  (Xbar0)
//   Router 1         : Crossbar partition 1  (Xbar1)
//   Router 2..K+1    : HBM routers 0..K-1   (HBM0..HBMK-1)
//
// Traffic matrix file format (tab-separated):
//   Header row: row_label <TAB> Xbar0 <TAB> Xbar1 <TAB> HBM0 <TAB> ... <TAB> HBM{K-1}
//   Data rows:  <router_name> <TAB> <MB> <TAB> ...
//   Sum rows (containing "sum") are skipped.

class MoETrafficManager : public BatchTrafficManager {

protected:

  vector<vector<int>> _moe_matrix;
  vector<vector<int>> _moe_remaining;
  vector<int> _moe_dest_rr;

  int _moe_total_packets;
  int _moe_received_packets;

  double _moe_total_mb;
  int _flit_width_bytes;

  void _LoadTrafficMatrix(const Configuration &config);

  virtual void _Inject() override;
  virtual bool _SingleSim() override;

public:

  MoETrafficManager(const Configuration &config, const vector<Network *> &net);
  virtual ~MoETrafficManager();

};

#endif
