from generate_preference_data import *

pref_types = {"impartial_culture": generate_impartial_culture_pref,
              "women_masterlist": generate_woman_masterlist_pref
}

def main(pref_type, n, trials, csv_name=None):
    return pref_types[pref_type](n, trials, csv_name)


# main("women_masterlist", 10, 500)
main("impartial_culture", 5, 500)