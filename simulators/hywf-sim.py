#!/usr/bin/env python3
import argparse, sys, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-d", required=True, default="", help="root folder of dataset")
ap.add_argument("-r", required=True, default="", help="folder to store results")
ap.add_argument("-s", required=True, default=10, type=int, help="number of samples per trace")
args = vars(ap.parse_args())

def main():
    if not os.path.isdir(args["d"]):
        sys.exit(f"{args['d']} is not a directory")
    if os.path.exists(args["r"]):
        sys.exit(f"{args['r']} already exits")
    os.makedirs(args["r"])

    for _, _, files in os.walk(args["d"]):
        for f in files:
            for sample in range(args["s"]):
                r, o = sim_hywf_trace(os.path.join(args["d"], f))
                base = os.path.join(args["r"], f)
                if sample == 0:
                    save(base, o)
                save(f"{base}-p0-s{sample}", normalize(r[0]))
                save(f"{base}-p1-s{sample}", normalize(r[1]))

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

def sim_hywf_trace(tracefname, cons=20, delim="\t"):
    # simulates the HyWF splitting, assuming the same alg (Alg 1 in paper) 
    # is running at both a client and the relay.
    # results[0] is from the first path, results[1] second path
    results = [[], []]
    orig = []

    def sample_limit():
        return np.random.geometric(1./cons)

    def select_interface(p):
        return 1 if np.random.uniform() < p else 0

    with open(tracefname, "r") as f:
        # we need to simulate the alg on client and relay
        p_client, p_relay = np.random.uniform(), np.random.uniform()
        n_client, n_relay = 0, 0
        c_client, c_relay = 0, 0
        i_client, i_relay = 0, 0

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
                    if n_client > c_client:
                        c_client = sample_limit()
                        i_client = select_interface(p_client)
                        n_client = 0

                    l = results[i_client]
                    l.append(line)
                    results[i_client] = l
                else:
                    # client is recv, so sent on one path from relay
                    n_relay += 1
                    if n_relay > c_relay:
                        c_relay = sample_limit()
                        i_relay = select_interface(p_relay)
                        n_relay = 0

                    l = results[i_relay]
                    l.append(line)
                    results[i_relay] = l
        except:
            print(f"fail for {tracefname}")

    return results, orig

if __name__ == "__main__":
    main()