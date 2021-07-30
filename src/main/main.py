import sys
import json
from glob import glob
from src.input.tidy import arguments
from src.input.parser import read
from src.model.all import learn
from src.model.all import train
from src.model.all import evaluate
from src.output.tabulate import mean
from src.output.csv import ascsv

if __name__ == "__main__":
    # Read input
    files = arguments(glob(sys.argv[1]))
    raws  = [read(o, s) for o, s in files]
    raws  = [r.fix() for r in raws]
    data  = [r.data() for r in raws]
    data  = [r.tokenize() for r in data]

    # Separate input
    originals  = [d.original for d in data]
    references = [d.sums for d in data]
    #originals  = [originals[0], originals[155]]
    #references = [references[0], references[155]]

    results, times = learn(originals, references)

    ## Train models
    #summaries, times = train(originals)

    ## Test models
    #results = evaluate(summaries, references, originals)

    # Write output
    #print(ascsv(originals, summaries, [0, 1]))
    print(json.dumps(mean(results), indent = 2))
    print(json.dumps(times, indent = 2))
