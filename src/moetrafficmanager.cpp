#include <sstream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <iostream>

#include "moetrafficmanager.hpp"
#include "hbmnet.hpp"
#include "random_utils.hpp"
#include "packet_reply_info.hpp"

MoETrafficManager::MoETrafficManager(const Configuration &config,
                                                     const vector<Network *> &net)
  : BatchTrafficManager(config, net),
    _moe_total_packets(0),
    _moe_received_packets(0)
{
  _flit_width_bytes = config.GetInt("flit_width_bytes");
  _moe_total_mb = config.GetFloat("moe_total_mb");

  _moe_dest_rr.resize(_nodes, 0);

  _LoadTrafficMatrix(config);
}

MoETrafficManager::~MoETrafficManager()
{
}

// Node range for a named source/destination in the traffic matrix.
struct NodeRange {
  int first;  // first node ID
  int count;  // number of nodes (1 for individual SM/L2, >1 for router-level)
};

// Returns true if name is a recognized Xbar/HBM/SM/L2 label.
static bool _IsValidName(const string &name)
{
  if (name.size() >= 4 && name.substr(0,4) == "Xbar") return true;
  if (name.size() >= 3 && name.substr(0,3) == "HBM")  return true;
  if (name.size() >= 3 && name.substr(0,2) == "SM")   return true;
  if (name.size() >= 2 && name.substr(0,2) == "L2")   return true;
  return false;
}

// Map a traffic-matrix name to a (first_node, count) range.
//   "SM{i}"         -> (i, 1)                                   individual SM node
//   "L2_{i}"/"L2{i}"-> (num_sms + i, 1)                        individual L2 slice node
//   "Xbar{p}"       -> (RouterFirstNode[p], RouterConc[p])      all SMs on that crossbar partition
//   "HBM{h}"        -> (RouterFirstNode[h+2], RouterConc[h+2])  all L2 slices on that HBM router
static NodeRange _ParseNodeRange(const string &name, int num_routers, int total_nodes)
{
  NodeRange bad = {-1, 0};

  if (name.size() >= 2 && name.substr(0,2) == "SM") {
    int i = atoi(name.substr(2).c_str());
    if (i < 0 || i >= total_nodes) return bad;
    return {i, 1};
  }
  if (name.size() >= 2 && name.substr(0,2) == "L2") {
    // num_sms = first node of the first HBM router (router 2)
    int num_sms = (num_routers > 2) ? gHBMNetRouterFirstNode[2] : 0;
    string rest = name.substr(2);
    if (!rest.empty() && rest[0] == '_') rest = rest.substr(1);
    int i = atoi(rest.c_str());
    int node = num_sms + i;
    if (node < 0 || node >= total_nodes) return bad;
    return {node, 1};
  }
  if (name.size() >= 4 && name.substr(0,4) == "Xbar") {
    int p = atoi(name.substr(4).c_str());
    if (p < 0 || p >= num_routers) return bad;
    return {gHBMNetRouterFirstNode[p], gHBMNetRouterConc[p]};
  }
  if (name.size() >= 3 && name.substr(0,3) == "HBM") {
    int h = atoi(name.substr(3).c_str());
    int rid = h + 2;
    if (rid < 0 || rid >= num_routers) return bad;
    return {gHBMNetRouterFirstNode[rid], gHBMNetRouterConc[rid]};
  }
  return bad;
}

static vector<string> _SplitTabs(const string &line)
{
  vector<string> tokens;
  istringstream iss(line);
  string token;
  while (getline(iss, token, '\t')) {
    size_t start = token.find_first_not_of(" \r\n");
    size_t end   = token.find_last_not_of(" \r\n");
    if (start != string::npos)
      tokens.push_back(token.substr(start, end - start + 1));
    else
      tokens.push_back("");
  }
  return tokens;
}

void MoETrafficManager::_LoadTrafficMatrix(const Configuration &config)
{
  int packet_size     = _GetNextPacketSize(0);
  int bytes_per_packet = packet_size * _flit_width_bytes;
  int num_routers     = _net[0]->NumRouters();

  _moe_matrix.assign(_nodes, vector<int>(_nodes, 0));
  _moe_remaining.assign(_nodes, vector<int>(_nodes, 0));

  string matrix_file = config.GetStr("traffic_matrix_file");

  if (matrix_file != "") {
    ifstream in(matrix_file.c_str());
    if (!in.is_open()) {
      cerr << "Error: Cannot open traffic matrix file: " << matrix_file << endl;
      exit(-1);
    }

    vector<string> lines;
    string line;
    while (getline(in, line)) {
      if (!line.empty()) lines.push_back(line);
    }
    in.close();

    cout << "MoE: Read " << lines.size() << " lines from " << matrix_file << endl;

    // Find header row: look for any Xbar/HBM/SM/L2 label in column 1+
    int header_row = -1;
    vector<string> col_names;
    for (size_t r = 0; r < lines.size(); ++r) {
      vector<string> tokens = _SplitTabs(lines[r]);
      bool found = false;
      for (size_t t = 1; t < tokens.size(); ++t) {
        if (_IsValidName(tokens[t])) { found = true; break; }
      }
      if (found) {
        header_row = (int)r;
        for (size_t t = 1; t < tokens.size(); ++t)
          if (_IsValidName(tokens[t])) col_names.push_back(tokens[t]);
        break;
      }
    }

    if (header_row < 0) {
      cerr << "Error: Could not find header row with node/router names "
              "(Xbar{p}, HBM{h}, SM{i}, L2_{i})." << endl;
      exit(-1);
    }

    cout << "MoE: Header at line " << header_row
         << ", " << col_names.size() << " dest columns: ";
    for (size_t i = 0; i < col_names.size(); ++i) cout << col_names[i] << " ";
    cout << endl;

    // Resolve each column name to a node range
    vector<NodeRange> col_ranges;
    for (size_t i = 0; i < col_names.size(); ++i) {
      NodeRange nr = _ParseNodeRange(col_names[i], num_routers, _nodes);
      cout << "MoE:   Column '" << col_names[i]
           << "' -> first=" << nr.first << " count=" << nr.count << endl;
      assert(nr.first >= 0 && nr.first + nr.count <= _nodes);
      col_ranges.push_back(nr);
    }

    // Parse data rows; distribute MB directly into _moe_matrix
    cout << "MoE: Distributing to nodes (bytes_per_packet=" << bytes_per_packet << "):" << endl;
    for (size_t r = header_row + 1; r < lines.size(); ++r) {
      vector<string> tokens = _SplitTabs(lines[r]);
      if (tokens.empty()) continue;

      string row_name = tokens[0];
      if (row_name.find("sum") != string::npos || row_name.find("Sum") != string::npos)
        continue;
      if (!_IsValidName(row_name)) {
        cout << "MoE:   Skipping row '" << row_name << "' (unknown name)" << endl;
        continue;
      }

      NodeRange src = _ParseNodeRange(row_name, num_routers, _nodes);
      if (src.first < 0) {
        cout << "MoE:   Skipping row '" << row_name << "' (invalid range)" << endl;
        continue;
      }

      cout << "MoE:   Row '" << row_name
           << "' (first=" << src.first << " count=" << src.count << "): ";

      int num_vals = min((int)col_ranges.size(), (int)tokens.size() - 1);
      for (int c = 0; c < num_vals; ++c) {
        double mb = atof(tokens[c + 1].c_str());
        if (mb <= 0.0) continue;

        NodeRange dst = col_ranges[c];

        // Count valid pairs (skip self-to-self)
        int valid_pairs = 0;
        for (int si = 0; si < src.count; ++si)
          for (int di = 0; di < dst.count; ++di)
            if (src.first + si != dst.first + di) ++valid_pairs;
        if (valid_pairs == 0) continue;

        double total_bytes = mb * 1024.0 * 1024.0;
        int total_pkts  = (int)round(total_bytes / (double)bytes_per_packet);
        int base        = total_pkts / valid_pairs;
        int remainder   = total_pkts % valid_pairs;
        int pair_idx    = 0;

        for (int si = 0; si < src.count; ++si) {
          int sn = src.first + si;
          for (int di = 0; di < dst.count; ++di) {
            int dn = dst.first + di;
            if (sn == dn) continue;
            _moe_matrix[sn][dn] += base + (pair_idx < remainder ? 1 : 0);
            ++pair_idx;
          }
        }

        cout << col_names[c] << "=" << mb << "MB"
             << "(pairs=" << valid_pairs << ",pkts=" << total_pkts << ") ";
      }
      cout << endl;
    }
  } else {
    // Default: all-to-all uniform across all router pairs
    for (int sr = 0; sr < num_routers; ++sr) {
      for (int dr = 0; dr < num_routers; ++dr) {
        if (sr == dr) continue;
        int src_conc  = gHBMNetRouterConc[sr];
        int dst_conc  = gHBMNetRouterConc[dr];
        int src_first = gHBMNetRouterFirstNode[sr];
        int dst_first = gHBMNetRouterFirstNode[dr];
        int num_pairs = src_conc * dst_conc;
        double total_bytes = _moe_total_mb * 1024.0 * 1024.0;
        int total_pkts = (int)round(total_bytes / (double)bytes_per_packet);
        int base_per_pair = total_pkts / num_pairs;
        int remainder     = total_pkts % num_pairs;
        int pair_idx = 0;
        for (int si = 0; si < src_conc; ++si)
          for (int di = 0; di < dst_conc; ++di) {
            _moe_matrix[src_first + si][dst_first + di] =
              base_per_pair + (pair_idx < remainder ? 1 : 0);
            ++pair_idx;
          }
      }
    }
  }

  // Count totals and init remaining
  _moe_total_packets = 0;
  for (int s = 0; s < _nodes; ++s)
    for (int d = 0; d < _nodes; ++d) {
      _moe_remaining[s][d] = _moe_matrix[s][d];
      _moe_total_packets += _moe_matrix[s][d];
    }

  cout << "MoE Traffic Matrix loaded:" << endl;
  cout << "  packet_size = " << packet_size << " flits (" << bytes_per_packet << " bytes)" << endl;
  cout << "  total_packets = " << _moe_total_packets << endl;

  cout << "  Per-node outgoing packets (first 20 active):" << endl;
  int shown = 0;
  for (int s = 0; s < _nodes && shown < 20; ++s) {
    int total = 0;
    for (int d = 0; d < _nodes; ++d) total += _moe_matrix[s][d];
    if (total > 0) {
      int rtr = hbmnet_node_to_router(s);
      cout << "    Node " << s << " (router " << rtr
           << ", port " << hbmnet_node_to_port(s) << "): "
           << total << " pkts total" << endl;
      ++shown;
    }
  }
}

void MoETrafficManager::_Inject()
{
  for (int source = 0; source < _nodes; ++source) {
    if (!_partial_packets[source][0].empty()) continue;

    int dest = -1;
    for (int i = 0; i < _nodes; ++i) {
      int d = (_moe_dest_rr[source] + i) % _nodes;
      if (_moe_remaining[source][d] > 0) {
        dest = d;
        _moe_dest_rr[source] = (d + 1) % _nodes;
        break;
      }
    }
    if (dest < 0) continue;

    _moe_remaining[source][dest]--;

    int size = _GetNextPacketSize(0);
    int pid  = _cur_pid++;
    assert(_cur_pid);
    bool watch = gWatchOut && (_packets_to_watch.count(pid) > 0);

    for (int i = 0; i < size; ++i) {
      Flit *f = Flit::New();
      f->id          = _cur_id++;
      assert(_cur_id);
      f->pid         = pid;
      f->watch       = watch | (gWatchOut && (_flits_to_watch.count(f->id) > 0));
      f->subnetwork  = 0;
      f->src         = source;
      f->ctime       = _time;
      f->record      = true;
      f->cl          = 0;
      f->type        = Flit::ANY_TYPE;

      _total_in_flight_flits[0].insert(make_pair(f->id, f));
      _measured_in_flight_flits[0].insert(make_pair(f->id, f));

      if (i == 0) {
        f->head = true;
        f->dest = dest;
      } else {
        f->head = false;
        f->dest = -1;
      }

      f->pri  = 0;
      f->tail = (i == size - 1);
      f->vc   = -1;

      _partial_packets[source][0].push_back(f);
    }
  }
}

bool MoETrafficManager::_SingleSim()
{
  _moe_total_packets = 0;
  for (int s = 0; s < _nodes; ++s) {
    _moe_dest_rr[s] = 0;
    for (int d = 0; d < _nodes; ++d) {
      _moe_remaining[s][d]  = _moe_matrix[s][d];
      _moe_total_packets   += _moe_matrix[s][d];
    }
  }

  hbmnet_reset_stats();

  _sim_state           = running;
  int start_time       = _time;
  _moe_received_packets = 0;

  cout << "MoE: Starting batch of " << _moe_total_packets
       << " packets at time " << start_time << "..." << endl;

  bool all_injected = false;
  while (true) {
    _Step();

    if (!all_injected) {
      all_injected = true;
      for (int s = 0; s < _nodes && all_injected; ++s)
        for (int d = 0; d < _nodes && all_injected; ++d)
          if (_moe_remaining[s][d] > 0) all_injected = false;
    }

    bool packets_left = false;
    for (int c = 0; c < _classes; ++c)
      packets_left |= !_total_in_flight_flits[c].empty();

    if (!packets_left)
      for (int s = 0; s < _nodes && !packets_left; ++s)
        for (int c = 0; c < _classes && !packets_left; ++c)
          packets_left |= !_partial_packets[s][c].empty();

    if (all_injected && !packets_left) break;

    if ((_time - start_time) % 10000 == 0) {
      int remaining = 0;
      for (int s = 0; s < _nodes; ++s)
        for (int d = 0; d < _nodes; ++d)
          remaining += _moe_remaining[s][d];
      int in_flight = 0;
      for (int c = 0; c < _classes; ++c)
        in_flight += _total_in_flight_flits[c].size();
      cout << "MoE: t=" << (_time - start_time)
           << " remaining_to_inject=" << remaining
           << " in_flight=" << in_flight << endl;
    }

    if ((_time - start_time) > 100000000) {
      cout << "MoE: Timeout!" << endl;
      return false;
    }
  }

  int completion_time = _time - start_time;

  cout << "======================================" << endl;
  cout << "MoE Batch Completion Time = " << completion_time << " cycles" << endl;
  cout << "  Total packets = " << _moe_total_packets << endl;
  if (_plat_stats[0]->NumSamples() > 0) {
    cout << "  Avg packet latency = " << _plat_stats[0]->Average() << endl;
    cout << "  Avg network latency = " << _nlat_stats[0]->Average() << endl;
    cout << "  Avg hops = " << _hop_stats[0]->Average() << endl;
  }
  // UGAL non-minimal detour rate (only meaningful when routing_function = ugal or hybrid+ugal)
  {
    long long total_ugal = gUGALMinDecisions + gUGALNonMinDecisions;
    if (total_ugal > 0) {
      double nonmin_pct = 100.0 * (double)gUGALNonMinDecisions / (double)total_ugal;
      cout << "  UGAL decisions: total=" << total_ugal
           << "  min=" << gUGALMinDecisions
           << "  non-min=" << gUGALNonMinDecisions
           << "  non-min ratio=" << nonmin_pct << "%" << endl;
    }
  }
  // Escape VC usage: fraction of flits on vc==0 at ejection
  if (gTotalEjects > 0) {
    double pct = 100.0 * (double)gEscapeVCEjects / (double)gTotalEjects;
    cout << "  Escape VC usage = " << gEscapeVCEjects << " / " << gTotalEjects
         << "  (" << pct << "%)" << endl;
  }
  cout << "======================================" << endl;

  // === Per-router traffic analysis ===
  cout << "=== Per-Router Injection/Ejection Analysis ===" << endl;
  int num_routers = _net[0]->NumRouters();

  // Use fabric distance table for transit approximation (covers all links)
  const vector<vector<int>> &dist = gHBMNetDistFabric;

  for (int rtr = 0; rtr < num_routers; ++rtr) {
    int first = gHBMNetRouterFirstNode[rtr];
    int conc  = gHBMNetRouterConc[rtr];

    long long total_inject = 0, total_eject = 0;
    for (int p = 0; p < conc; ++p) {
      int node = first + p;
      long long node_send = 0, node_recv = 0;
      for (int d = 0; d < _nodes; ++d) node_send += _moe_matrix[node][d];
      for (int s = 0; s < _nodes; ++s) node_recv += _moe_matrix[s][node];
      total_inject += node_send;
      total_eject  += node_recv;
    }

    double inject_rate = (double)total_inject / (double)completion_time;
    double eject_rate  = (double)total_eject  / (double)completion_time;

    long long transit_pkts = 0;
    if (!dist.empty()) {
      for (int sr = 0; sr < num_routers; ++sr) {
        for (int dr = 0; dr < num_routers; ++dr) {
          if (sr == dr || sr == rtr || dr == rtr) continue;
          if (dist[sr][rtr] + dist[rtr][dr] == dist[sr][dr]) {
            int sf = gHBMNetRouterFirstNode[sr], sc = gHBMNetRouterConc[sr];
            int df = gHBMNetRouterFirstNode[dr], dc = gHBMNetRouterConc[dr];
            for (int si = 0; si < sc; ++si)
              for (int di = 0; di < dc; ++di)
                transit_pkts += _moe_matrix[sf + si][df + di];
          }
        }
      }
    }
    double transit_rate = (double)transit_pkts / (double)completion_time;

    cout << "  Router " << rtr
         << " (conc=" << conc << "):"
         << " inject=" << total_inject << " (" << inject_rate << " pkt/cyc)"
         << " eject=" << total_eject  << " (" << eject_rate  << " pkt/cyc)"
         << " transit~=" << transit_pkts << " (" << transit_rate << " pkt/cyc)"
         << " total_load~=" << (inject_rate + eject_rate + transit_rate) << " pkt/cyc"
         << endl;
  }
  cout << "===============================================" << endl;

  _batch_time->AddSample(completion_time);
  _drain_time  = _time;
  _reset_time  = start_time;

  return true;
}
