import itertools


# This function prints the number of combinations and returns a list of tuples of every possible combination of the
# list pass in as an argument.

def combos(lst):
    lst_combs = []
    for n in range(len(lst)-1, len(lst) + 1):
        lst_combs += list(itertools.combinations(lst, n))
    print(len(lst_combs))
    return lst_combs
