"""
to_spider.py -- write a topology in the Spider simulator's own input format, plus the OMNeT++ .ned files.

Topology file (see ../sample-topology.txt and hostInitialize.cc):
  line 1 : landmark routers ("<id>r ..."), only used by landmarkRouting
  router channels : "<a>r <b>r <delay_ms a->b> <delay_ms b->a> <balance_a> <balance_b>"
  end hosts       : "<n>e <n>r <delay> <delay> <big> <big>"     (one end host per router, huge balance)
Delays MUST be integers (the C++ parser uses stoi). Node ids must be 0..N-1.
"""
import os

LARGE_BALANCE = 1000000000          # same constant as scripts/config.py
NUM_LANDMARKS = 4

# routing scheme -> (host module, router module); mirrors scripts/create_topo_ned_file.py
def _types(alg):
    if alg == "shortestPath":
        return "hostNodeBase", "routerNodeBase"
    host = "hostNode" + alg[0].upper() + alg[1:]
    router = "routerNodeWaterfilling" if alg == "landmarkRouting" else "routerNode" + alg[0].upper() + alg[1:]
    return host, router


def write_spider_topology(G, path, delay_ms=30, balance_split=0.5):
    """G: nx.Graph with nodes 0..N-1 and edge attr 'cap' (total channel capacity)."""
    n = G.number_of_nodes()
    assert sorted(G.nodes()) == list(range(n)), "nodes must be 0..N-1"
    landmarks = [v for v, _ in sorted(G.degree(), key=lambda x: -x[1])[:NUM_LANDMARKS]]
    with open(path, "w") as f:
        f.write(" ".join(f"{v}r" for v in landmarks) + " \n")
        for u, v, d in G.edges(data=True):
            cap = max(2, int(round(d["cap"])))
            b_u = int(round(cap * balance_split))
            f.write(f"{u}r {v}r {int(delay_ms)} {int(delay_ms)} {b_u} {cap - b_u}\n")
        for v in range(n):
            f.write(f"{v}e {v}r {int(delay_ms)} {int(delay_ms)} {LARGE_BALANCE // 2} {LARGE_BALANCE // 2}\n")


def write_ned(topo_path, ned_path, network_name, routing_alg):
    """Same output as write_ned_file() in scripts/create_topo_ned_file.py."""
    def parse(name):
        return ("router[%d]" % int(name[:-1]), int(name[:-1]), "r") if name[-1] == "r" else \
               ("host[%d]" % int(name[:-1]), int(name[:-1]), "e")
    links, max_r, max_h = [], -1, -1
    with open(topo_path) as f:
        for i, line in enumerate(f):
            if i == 0 or not line.strip():
                continue
            t = line.split()
            a, b = parse(t[0]), parse(t[1])
            for x in (a, b):
                if x[2] == "r":
                    max_r = max(max_r, x[1])
                else:
                    max_h = max(max_h, x[1])
            links.append((a[0], b[0], float(t[2]), float(t[3])))
    host_t, router_t = _types(routing_alg)
    with open(ned_path, "w") as o:
        o.write(f"import {router_t};\nimport {host_t};\n\n")
        o.write(f"network {network_name}_{routing_alg}\n{{\n")
        o.write('\tparameters:\n\t\tdouble linkDelay @unit("s") = default(100us);\n')
        o.write('\t\tdouble linkDataRate @unit("Gbps") = default(1Gbps);\n\tsubmodules:\n')
        o.write(f"\t\thost[{max_h + 1}]: {host_t} {{}} \n\t\trouter[{max_r + 1}]: {router_t} {{}} \n\tconnections: \n")
        for a, b, ab, ba in links:
            o.write(f"\t\t{a}.out++ --> {{delay = {ab}ms; }} --> {b}.in++;  \n")
            o.write(f"\t\t{a}.in++ <-- {{delay = {ba}ms; }} <-- {b}.out++;  \n")
        o.write("}\n")
