// $Id$

/*
 Copyright (c) 2007-2015, Trustees of The Leland Stanford Junior University
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this 
 list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <iostream>
#include <sstream>
#include <ctime>
#include "random_utils.hpp"
#include "traffic.hpp"

TrafficPattern::TrafficPattern(int nodes)
: _nodes(nodes)
{
  if(nodes <= 0) {
    cout << "Error: Traffic patterns require at least one node." << endl;
    exit(-1);
  }
}

void TrafficPattern::reset()
{

}

TrafficPattern * TrafficPattern::New(string const & pattern, int nodes, 
				     Configuration const * const config)
{
  string pattern_name;
  string param_str;
  size_t left = pattern.find_first_of('(');
  if(left == string::npos) {
    pattern_name = pattern;
  } else {
    pattern_name = pattern.substr(0, left);
    size_t right = pattern.find_last_of(')');
    if(right == string::npos) {
      param_str = pattern.substr(left+1);
    } else {
      param_str = pattern.substr(left+1, right-left-1);
    }
  }
  vector<string> params = tokenize_str(param_str);
  
  TrafficPattern * result = NULL;
  if(pattern_name == "bitcomp") {
    result = new BitCompTrafficPattern(nodes);
  } else if(pattern_name == "transpose") {
    result = new TransposeTrafficPattern(nodes);
  } else if(pattern_name == "bitrev") {
    result = new BitRevTrafficPattern(nodes);
  } else if(pattern_name == "shuffle") {
    result = new ShuffleTrafficPattern(nodes);
  } else if(pattern_name == "randperm") {
    int perm_seed = -1;
    if(params.empty()) {
      if(config) {
	if(config->GetStr("perm_seed") == "time") {
	  perm_seed = int(time(NULL));
	  cout << "SEED: perm_seed=" << perm_seed << endl;
	} else {
	  perm_seed = config->GetInt("perm_seed");
	}
      } else {
	cout << "Error: Missing parameter for random permutation traffic pattern: " << pattern << endl;
	exit(-1);
      }
    } else {
      perm_seed = atoi(params[0].c_str());
    }
    result = new RandomPermutationTrafficPattern(nodes, perm_seed);
  } else if(pattern_name == "uniform") {
    result = new UniformRandomTrafficPattern(nodes);
  } else if(pattern_name == "background") {
    vector<int> excludes = tokenize_int(params[0]);
    result = new UniformBackgroundTrafficPattern(nodes, excludes);
  } else if(pattern_name == "diagonal") {
    result = new DiagonalTrafficPattern(nodes);
  } else if(pattern_name == "asymmetric") {
    result = new AsymmetricTrafficPattern(nodes);
  } else if(pattern_name == "taper64") {
    result = new Taper64TrafficPattern(nodes);
  } else if(pattern_name == "bad_dragon") {
    bool missing_params = false;
    int k = -1;
    if(params.size() < 1) {
      if(config) {
	k = config->GetInt("k");
      } else {
	missing_params = true;
      }
    } else {
      k = atoi(params[0].c_str());
    }
    int n = -1;
    if(params.size() < 2) {
      if(config) {
	n = config->GetInt("n");
      } else {
	missing_params = true;
      }
    } else {
      n = atoi(params[1].c_str());
    }
    if(missing_params) {
      cout << "Error: Missing parameters for dragonfly bad permutation traffic pattern: " << pattern << endl;
      exit(-1);
    }
    result = new BadPermDFlyTrafficPattern(nodes, k, n);
  } else if((pattern_name == "tornado") || (pattern_name == "neighbor") ||
	    (pattern_name == "badperm_yarc")) {
    bool missing_params = false;
    int k = -1;
    if(params.size() < 1) {
      if(config) {
	k = config->GetInt("k");
      } else {
	missing_params = true;
      }
    } else {
      k = atoi(params[0].c_str());
    }
    int n = -1;
    if(params.size() < 2) {
      if(config) {
	n = config->GetInt("n");
      } else {
	missing_params = true;
      }
    } else {
      n = atoi(params[1].c_str());
    }
    int xr = -1;
    if(params.size() < 3) {
      if(config) {
	xr = config->GetInt("xr");
      } else {
	missing_params = true;
      }
    } else {
      xr = atoi(params[2].c_str());
    }
    if(missing_params) {
      cout << "Error: Missing parameters for digit permutation traffic pattern: " << pattern << endl;
      exit(-1);
    }
    if(pattern_name == "tornado") {
      result = new TornadoTrafficPattern(nodes, k, n, xr);
    } else if(pattern_name == "neighbor") {
      result = new NeighborTrafficPattern(nodes, k, n, xr);
    } else if(pattern_name == "badperm_yarc") {
      result = new BadPermYarcTrafficPattern(nodes, k, n, xr);
    }
  } else if(pattern_name == "gpu") {
    result = new GPUTrafficPattern(nodes, config);
  } else if(pattern_name == "hotspot") {
    if(params.empty()) {
      params.push_back("-1");
    } 
    vector<int> hotspots = tokenize_int(params[0]);
    for(size_t i = 0; i < hotspots.size(); ++i) {
      if(hotspots[i] < 0) {
	hotspots[i] = RandomInt(nodes - 1);
      }
    }
    vector<int> rates;
    if(params.size() >= 2) {
      rates = tokenize_int(params[1]);
      rates.resize(hotspots.size(), rates.back());
    } else {
      rates.resize(hotspots.size(), 1);
    }
    result = new HotSpotTrafficPattern(nodes, hotspots, rates);
  } else {
    cout << "Error: Unknown traffic pattern: " << pattern << endl;
    exit(-1);
  }
  return result;
}

PermutationTrafficPattern::PermutationTrafficPattern(int nodes)
  : TrafficPattern(nodes)
{
  
}

BitPermutationTrafficPattern::BitPermutationTrafficPattern(int nodes)
  : PermutationTrafficPattern(nodes)
{
  if((nodes & -nodes) != nodes) {
    cout << "Error: Bit permutation traffic patterns require the number of "
	 << "nodes to be a power of two." << endl;
    exit(-1);
  }
}

BitCompTrafficPattern::BitCompTrafficPattern(int nodes)
  : BitPermutationTrafficPattern(nodes)
{
  
}

int BitCompTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes));
  int const mask = _nodes - 1;
  return ~source & mask;
}

TransposeTrafficPattern::TransposeTrafficPattern(int nodes)
  : BitPermutationTrafficPattern(nodes), _shift(0)
{
  while(nodes >>= 1) {
    ++_shift;
  }
  if(_shift % 2) {
    cout << "Error: Transpose traffic pattern requires the number of nodes to "
	 << "be an even power of two." << endl;
    exit(-1);
  }
  _shift >>= 1;
}

int TransposeTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes));
  int const mask_lo = (1 << _shift) - 1;
  int const mask_hi = mask_lo << _shift;
  return (((source >> _shift) & mask_lo) | ((source << _shift) & mask_hi));
}

BitRevTrafficPattern::BitRevTrafficPattern(int nodes)
  : BitPermutationTrafficPattern(nodes)
{
  
}

int BitRevTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes));
  int result = 0;
  for(int n = _nodes; n > 1; n >>= 1) {
    result = (result << 1) | (source % 2);
    source >>= 1;
  }
  return result;
}

ShuffleTrafficPattern::ShuffleTrafficPattern(int nodes)
  : BitPermutationTrafficPattern(nodes)
{

}

int ShuffleTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes));
  int const shifted = source << 1;
  return ((shifted & (_nodes - 1)) | bool(shifted & _nodes));
}

DigitPermutationTrafficPattern::DigitPermutationTrafficPattern(int nodes, int k,
							       int n, int xr)
  : PermutationTrafficPattern(nodes), _k(k), _n(n), _xr(xr)
{
  
}

TornadoTrafficPattern::TornadoTrafficPattern(int nodes, int k, int n, int xr)
  : DigitPermutationTrafficPattern(nodes, k, n, xr)
{

}

int TornadoTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes));

  int offset = 1;
  int result = 0;

  for(int n = 0; n < _n; ++n) {
    result += offset *
      (((source / offset) % (_xr * _k) + ((_xr * _k + 1) / 2 - 1)) % (_xr * _k));
    offset *= (_xr * _k);
  }
  return result;
}

NeighborTrafficPattern::NeighborTrafficPattern(int nodes, int k, int n, int xr)
  : DigitPermutationTrafficPattern(nodes, k, n, xr)
{

}

int NeighborTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes));

  int offset = 1;
  int result = 0;
  
  for(int n = 0; n < _n; ++n) {
    result += offset *
      (((source / offset) % (_xr * _k) + 1) % (_xr * _k));
    offset *= (_xr * _k);
  }
  return result;
}

RandomPermutationTrafficPattern::RandomPermutationTrafficPattern(int nodes, 
								 int seed)
  : TrafficPattern(nodes)
{
  _dest.resize(nodes);
  randomize(seed);
}

void RandomPermutationTrafficPattern::randomize(int seed)
{
  vector<long> save_x;
  vector<double> save_u;
  SaveRandomState(save_x, save_u);
  RandomSeed(seed);

  _dest.assign(_nodes, -1);

  for(int i = 0; i < _nodes; ++i) {
    int ind = RandomInt(_nodes - 1 - i);

    int j = 0;
    int cnt = 0;
    while((cnt < ind) || (_dest[j] != -1)) {
      if(_dest[j] == -1) {
	++cnt;
      }
      ++j;
      assert(j < _nodes);
    }

    _dest[j] = i;
  }

  RestoreRandomState(save_x, save_u); 
}

int RandomPermutationTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes));
  assert((_dest[source] >= 0) && (_dest[source] < _nodes));
  return _dest[source];
}

RandomTrafficPattern::RandomTrafficPattern(int nodes)
  : TrafficPattern(nodes)
{

}

UniformRandomTrafficPattern::UniformRandomTrafficPattern(int nodes)
  : RandomTrafficPattern(nodes)
{

}

int UniformRandomTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes));
  return RandomInt(_nodes - 1);
}

SMtoL2TrafficPattern::SMtoL2TrafficPattern(int nodes, int num_sms, int num_l2)
  : RandomTrafficPattern(nodes), _num_sms(num_sms), _num_l2(num_l2)
{
  assert(num_sms + num_l2 == nodes);
}

int SMtoL2TrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes));
  if (source < _num_sms) {
    // SM node: send to random L2 node
    return RandomInt(_num_l2 - 1) + _num_sms;
  } else {
    // L2 node: send to random SM node
    return RandomInt(_num_sms - 1);
  }
  // SM nodes: [0, _num_sms - 1]
  // L2 nodes: [_num_sms, _num_sms + _num_l2 - 1]
  return _num_sms + RandomInt(_num_l2 - 1);
}

UniformBackgroundTrafficPattern::UniformBackgroundTrafficPattern(int nodes, vector<int> excluded_nodes)
  : RandomTrafficPattern(nodes)
{
  for(size_t i = 0; i < excluded_nodes.size(); ++i) {
    int const node = excluded_nodes[i];
    assert((node >= 0) && (node < _nodes));
    _excluded.insert(node);
  }
}

int UniformBackgroundTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes));

  int result;

  do {
    result = RandomInt(_nodes - 1);
  } while(_excluded.count(result) > 0);

  return result;
}

DiagonalTrafficPattern::DiagonalTrafficPattern(int nodes)
  : RandomTrafficPattern(nodes)
{

}

int DiagonalTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes));
  return ((RandomInt(2) == 0) ? ((source + 1) % _nodes) : source);
}

AsymmetricTrafficPattern::AsymmetricTrafficPattern(int nodes)
  : RandomTrafficPattern(nodes)
{

}

int AsymmetricTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes));
  int const half = _nodes / 2;
  return (source % half) + (RandomInt(1) ? half : 0);
}

Taper64TrafficPattern::Taper64TrafficPattern(int nodes)
  : RandomTrafficPattern(nodes)
{
  if(nodes != 64) {
    cout << "Error: Tthe Taper64 traffic pattern requires the number of nodes "
	 << "to be exactly 64." << endl;
    exit(-1);
  }
}

int Taper64TrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes));
  if(RandomInt(1)) {
    return ((64 + source + 8 * (RandomInt(2) - 1) + (RandomInt(2) - 1)) % 64);
  } else {
    return RandomInt(_nodes - 1);
  }
}

BadPermDFlyTrafficPattern::BadPermDFlyTrafficPattern(int nodes, int k, int n)
  : DigitPermutationTrafficPattern(nodes, k, n, 1)
{
  
}

int BadPermDFlyTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes));

  int const grp_size_routers = 2 * _k;
  int const grp_size_nodes = grp_size_routers * _k;

  return ((RandomInt(grp_size_nodes - 1) + ((source / grp_size_nodes) + 1) * grp_size_nodes) % _nodes);
}

BadPermYarcTrafficPattern::BadPermYarcTrafficPattern(int nodes, int k, int n, 
						     int xr)
  : DigitPermutationTrafficPattern(nodes, k, n, xr)
{

}

int BadPermYarcTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes));
  int const row = source / (_xr * _k);
  return RandomInt((_xr * _k) - 1) * (_xr * _k) + row;
}

HotSpotTrafficPattern::HotSpotTrafficPattern(int nodes, vector<int> hotspots, 
					     vector<int> rates)
  : TrafficPattern(nodes), _hotspots(hotspots), _rates(rates), _max_val(-1)
{
  assert(!_hotspots.empty());
  size_t const size = _hotspots.size();
  _rates.resize(size, _rates.empty() ? 1 : _rates.back());
  for(size_t i = 0; i < size; ++i) {
    int const hotspot = _hotspots[i];
    assert((hotspot >= 0) && (hotspot < _nodes));
    int const rate = _rates[i];
    assert(rate > 0);
    _max_val += rate;
  }
}

int HotSpotTrafficPattern::dest(int source)
{
  assert((source >= 0) && (source < _nodes));

  if(_hotspots.size() == 1) {
    return _hotspots[0];
  }

  int pct = RandomInt(_max_val);

  for(size_t i = 0; i < (_hotspots.size() - 1); ++i) {
    int const limit = _rates[i];
    if(limit > pct) {
      return _hotspots[i];
    } else {
      pct -= limit;
    }
  }
  assert(_rates.back() > pct);
  return _hotspots.back();
}

// GPU Traffic Pattern: SM nodes send requests to L2 slices only
GPUTrafficPattern::GPUTrafficPattern(int nodes, Configuration const * const config)
   : TrafficPattern(nodes)
{
  _P = config->GetInt("num_xbars");
  _H = config->GetInt("hbm_per_side");
  _K = _P * _H * 2;
  _sm_per_xbar = config->GetInt("sm_per_xbar");
  _l2_per_hbm = config->GetInt("l2_per_hbm");
  _num_sms = _sm_per_xbar * _P;
  _num_l2_slices = _l2_per_hbm * _K;

  _remote_only = config->GetInt("remote_only");
}

int GPUTrafficPattern::dest(int source)
{
  if (source >= _num_sms) {
      return -1;
  }

  if (_remote_only) {
    // SM belongs to partition cur_partition (0..P-1).
    // Select an L2 slice that does NOT belong to this partition.
    //
    // HBM stack h belongs to partition: (h % (K/2)) / H
    // Each partition owns H*2 HBM stacks → H*2*_l2_per_hbm L2 slices.
    // Remote L2 slices = total - local = _num_l2_slices - H*2*_l2_per_hbm.
    int cur_partition = source / _sm_per_xbar;
    int local_l2 = _H * 2 * _l2_per_hbm;
    int remote_l2 = _num_l2_slices - local_l2;
    assert(remote_l2 > 0);

    // Pick a random index among remote L2 slices
    int idx = RandomInt(remote_l2 - 1);

    // Map idx to an actual L2 node by skipping local-partition slices
    for (int h = 0; h < _K; h++) {
      int part = (h % (_K / 2)) / _H;
      if (part == cur_partition) continue;
      if (idx < _l2_per_hbm)
        return _num_sms + h * _l2_per_hbm + idx;
      idx -= _l2_per_hbm;
    }
    assert(false);
    return -1;
  }
  else {
    return _num_sms + RandomInt(_num_l2_slices - 1);
  }
}