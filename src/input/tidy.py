import os

def arguments(files):
    """
    Tidy files so that we get [(original, sum1, sum2)].
    All files must be in the same directory.
    All files must have the notation of the corpus, i.e.
    - <ID>.txt
    - SUM_<ID>_<NAME1>.sum
    - SUM_<ID>_<NAME2>.sum

    where <ID> is the id of the document and <NAME1> <NAME2> are the
    names of the authors of the summary.

    - files: [str] with files following the notation.
    - return: [(str, [str])] where [(original, [summary])].
    """
    # Get just .sum
    sums = [x for x in files if x.endswith(".sum")]
    directory = "" if not files else os.path.dirname(files[0])
    return regenerate(directory, associate_id_names(sums))

# Dont' use outside of this module
def associate_id_names(files):
    """
    Associate for all IDs, their two NAMEs.

    - files: [str] with only *.sum files.
    - return: {str: [str]} where {id: [names]}.
    """
    association = {}
    for filename in files:
        ID, name = parts(filename)

        if ID in association:
            association[ID].append(name)
        else:
            association[ID] = [name]
    
    return association

def parts(filename):
    part = filename.split("_")
    ID = part[1] + "_" + part[2]

    part = part[3].split(".")
    name = part[0]

    return ID, name

def regenerate(directory, association):
    """
    Regenerate filenames from IDs and NAMEs

    - association: {str: [str]} where {id: [names]}.
    - return: [(str, [str])] where [(original, [summary])].
    """
    files = []
    for ID, names in association.items():
        original = os.path.join(directory, ID + ".txt")
        sums = [os.path.join(directory, "SUM_"+ID+"_"+n+".sum") for n in names]
        files.append((original, sums))

    return files
