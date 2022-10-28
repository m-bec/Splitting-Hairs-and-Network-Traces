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
                r, o = sim_comps_rw_trace(os.path.join(args["d"], f))
                base = os.path.join(args["r"], f)
                if sample == 0:
                    save(base, o)
                save(f"{base}-p0-s{sample}", normalize(r[0]))
                save(f"{base}-p1-s{sample}", normalize(r[1]))
                save(f"{base}-p2-s{sample}", normalize(r[2]))

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

def sim_comps_rw_trace(tracefname, cons=20, delim="\t", duration=0.1):
    # simulates the CoMPS real-world splitting strategy, as implemented and
    # described in the paper.
    #
    # results[0] is from the first path, results[1] second path, results [2]
    # third path
    results = [[], [], []]
    # we also return the original trace, cause lazy, and want to save it *for
    # training* (enforced in the csv files)
    orig = []

    def samplePaths(n=3):
        return np.random.dirichlet([1]*n)
    
    def selectPath(p, n=3):
        return np.random.choice(np.arange(0,n),p=p)

    with open(tracefname, "r") as f:
        # sample probabilities for paths
        p_client, p_middle = samplePaths(), samplePaths()

        # "i" for "interface"
        i_client, i_middle = selectPath(p_client), selectPath(p_middle)

        # dummy starting time
        t_client, t_middle = -1*duration, -1*duration

        try:
            for line in f:
                orig.append(line)
                parts = line.split(delim)
                if len(parts) != 2:
                    sys.exit(f"malformed trace: {parts}")
                time = float(parts[0])
                direction = int(parts[1])

                if direction > 0:
                    # client is sending, will send on one path

                    if time - t_client >= duration:
                        i_client = selectPath(p_client)
                        t_client = time

                    l = results[i_client]
                    l.append(line)
                    results[i_client] = l
                else:
                    # client is recv, so sent on one path from middle
                    if time - t_middle >= duration:
                        i_middle = selectPath(p_middle)
                        t_middle = time

                    l = results[i_middle]
                    l.append(line)
                    results[i_middle] = l
                    
        except (RuntimeError, TypeError, NameError) as err:
            print(f"fail for {tracefname}: {err}")

    return results, orig

if __name__ == "__main__":
    main()