import sys
import json
from glob import glob
from src.input.parser_mlsum import read
from src.model.all import learn
from src.output.tabulate import mean

if __name__ == "__main__":
    # Read input
    raw_train = read(sys.argv[1], 500)
    #raw_valid = read(sys.argv[2], 1)
    #raw_test  = read(sys.argv[3], 1)

    raws = [r.fix_mlsum() for r in raw_train]
    raws = [r.fix() for r in raws]
    data = [r.data() for r in raws]
    data = [d.tokenize() for d in data]

    # Separate input
    originals  = [d.original for d in data]
    references = [d.sums for d in data]

    # Learn
    results, times = learn(originals, references)

    # Write output
    print(json.dumps(mean(results), indent = 2))
    print(json.dumps(times, indent = 2))
