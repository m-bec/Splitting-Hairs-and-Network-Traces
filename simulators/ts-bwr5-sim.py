#!/usr/bin/env python3
import argparse, sys, os
from xmlrpc.client import Boolean
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-d", required=True, default="", help="root folder of dataset")
ap.add_argument("-r", required=True, default="", help="folder to store results")
ap.add_argument("-s", required=True, default=10, type=int, help="number of samples per trace")
ap.add_argument("-a", required=False, type=Boolean, default=False,
    help="use the actual version of BWR-5, not the paper version")
args = vars(ap.parse_args())

def main():
    if not os.path.isdir(args["d"]):
        sys.exit(f"{args['d']} is not a directory")
    if os.path.exists(args["r"]):
        sys.exit(f"{args['r']} already exits")
    os.makedirs(args["r"])

    simfunc = sim_ts_bwr5_trace
    if args["a"]:
        print("simulating the actual version of BWR-5")
        simfunc = sim_ts_bwr5_trace_actual
    else:
        print("simulating the paper version of BWR-5")
        simfunc = sim_ts_bwr5_trace

    for _, _, files in os.walk(args["d"]):
        for f in files:
            for sample in range(args["s"]):
                r, o = simfunc(os.path.join(args["d"], f))
                base = os.path.join(args["r"], f)
                if sample == 0:
                    save(base, o)
                save(f"{base}-p0-s{sample}", normalize(r[0]))
                save(f"{base}-p1-s{sample}", normalize(r[1]))
                save(f"{base}-p2-s{sample}", normalize(r[2]))
                save(f"{base}-p3-s{sample}", normalize(r[3]))
                save(f"{base}-p4-s{sample}", normalize(r[4]))

def save(fname, trace):
        with open(fname, "w") as f:
            for l in trace:
                f.write(l)

def normalize(trace, delim="\t"):
    done = []
    first = -1
    for t in trace:
        parts = t.split(delim)
        if len(parts) != 2:
            sys.exit(f"malformed trace: {parts}")
    
        time = float(parts[0])
        direction = int(parts[1])
    
        if first == -1:
            first = time
    
        # enforce time resultion of .1 ms
        done.append(f"{time-first:.5f}{delim}{direction}\n")
    
    return done

def sim_ts_bwr5_trace(tracefname, cons=20, delim="\t"):
    # simulates the BWR5 splitting strategy, *as described in the paper*,
    # assuming running independently at both client and middle.
    #
    # results[0] is from the first path, results[1] second path, etc
    results = [[], [], [], [], []]
    # we also return the original trace, cause lazy, and want to save it *for
    # training* (enforced in the csv files)
    orig = []

    def sampleBatch(low=50, high=70):
        return np.random.randint(low, high+1)

    def samplePaths(n=5):
        return np.random.dirichlet([1]*n)
    
    def selectPath(p, n=5):
        return np.random.choice(np.arange(0,n),p=p)

    with open(tracefname, "r") as f:
        # sample probabilities for paths
        p_client, p_middle = samplePaths(), samplePaths()

        # "i" for "interface"
        i_client, i_middle = selectPath(p_client), selectPath(p_middle)

        # sampled batches and respective counters
        b_client, b_middle = sampleBatch(), sampleBatch()
        n_client, n_middle = 0, 0

        try:
            for line in f:
                orig.append(line)
                parts = line.split(delim)
                if len(parts) != 2:
                    sys.exit(f"malformed trace: {parts}")
                time = parts[0]
                direction = int(parts[1])

                if direction > 0:
                    # client is sending, will send on one path
                    n_client += 1
                    if n_client > b_client:
                        b_client = sampleBatch()
                        i_client = selectPath(p_client)
                        n_client = 0

                    l = results[i_client]
                    l.append(line)
                    results[i_client] = l
                else:
                    # client is recv, so sent on one path from middle
                    n_middle += 1
                    if n_middle > b_middle:
                        b_middle = sampleBatch()
                        i_middle = selectPath(p_middle)
                        n_middle = 0

                    l = results[i_middle]
                    l.append(line)
                    results[i_middle] = l
                    
        except:
            print(f"fail for {tracefname}")

    return results, orig

def sim_ts_bwr5_trace_actual(tracefname, cons=20, delim="\t"):
    # simulates the BWR5 splitting strategy, *as implemented in Tor and the
    # simulator by the authors*, assuming running independently at both client
    # and middle.
    #
    # results[0] is from the first path, results[1] second path, etc
    results = [[], [], [], [], []]
    # we also return the original trace, cause lazy, and want to save it *for
    # training* (enforced in the csv files)
    orig = []

    def sampleBatch(low=50, high=70):
        return np.random.randint(low, high+1)

    def samplePaths(n=5):
        return np.random.dirichlet([1]*n)

    def selectPath(p, n=5):
        return np.random.choice(np.arange(0,n),p=p)

    with open(tracefname, "r") as f:
        # sample probabilities for paths
        p_client, p_middle = samplePaths(), samplePaths()

        # "i" for "interface"
        i_client, i_middle = selectPath(p_client), selectPath(p_middle)

        # sampled batches and respective counters
        n_client, n_middle = 0, 0

        try:
            for line in f:
                orig.append(line)
                parts = line.split(delim)
                if len(parts) != 2:
                    sys.exit(f"malformed trace: {parts}")
                time = parts[0]
                direction = int(parts[1])

                if direction > 0:
                    # client is sending, will send on one path
                    n_client += 1
                    if n_client % sampleBatch() == 0:
                        i_client = selectPath(p_client)

                    l = results[i_client]
                    l.append(line)
                    results[i_client] = l
                else:
                    # client is recv, so sent on one path from middle
                    n_middle += 1
                    if n_middle % sampleBatch() == 0:
                        i_middle = selectPath(p_middle)

                    l = results[i_middle]
                    l.append(line)
                    results[i_middle] = l

        except:
            print(f"fail for {tracefname}")

    return results, orig

if __name__ == "__main__":
    main()