from collections import namedtuple
from diplomacy import Map

GamesStats = namedtuple('GamesStats', ['nb_won', 'nb_most', 'nb_survived', 'nb_defeated',
                                        'nb_power_assignations', 'assigned_powers', 'rankings'])

def get_stats(games):
    """ Computes stats """
    nb_won, nb_most, nb_survived, nb_defeated = 0, 0, 0, 0
    nb_power_assignations = {power_name: 0 for power_name in Map().powers}
    players_names = next(iter(games))['players_names']
    assigned_powers, rankings = [], []

    for game in games:
        if not game:
            continue

        game_assigned_powers = game['assigned_powers']
        nb_centers = {power_name: len(game['phases'][-1]['state']['centers'][power_name])
                      for power_name in game_assigned_powers}
        if nb_centers[game_assigned_powers[0]] >= 18:
            nb_won += 1
        elif nb_centers[game_assigned_powers[0]] == max(nb_centers.values()):
            nb_most += 1
        elif nb_centers[game_assigned_powers[0]] > 0:
            nb_survived += 1
        else:
            nb_defeated += 1

        nb_power_assignations[game_assigned_powers[0]] += 1
        assigned_powers.append(game_assigned_powers)
        rankings.append(game['ranking'])

    return players_names, GamesStats(nb_won, nb_most, nb_survived, nb_defeated, \
                                     nb_power_assignations, assigned_powers, rankings)

def print_ranking_stats(benchmark_name, games):
    nb_games = len(games)
    nb_completed_games = len([_ for _ in games if _ is not None])
    
    if not nb_completed_games:
        print('-' * 80)
        print('Benchmark: %s (%d/%d games)' % (benchmark_name, nb_completed_games, nb_games))
        print()
        print('No games to get stats from.')
        print('-' * 80)
        return

    # Computing stats
    players_names, stats = get_stats(games)

    # Displaying stats
    print('-' * 80)
    print('Benchmark: %s (%d/%d games)' % (benchmark_name, nb_completed_games, nb_games))
    print()
    print('Games Won: (%d) (%.2f)' % (stats.nb_won, 100. * stats.nb_won / nb_completed_games))
    print('Games Most SC: (%d) (%.2f)' % (stats.nb_most, 100. * stats.nb_most / nb_completed_games))
    print('Games Survived: (%d) (%.2f)' % (stats.nb_survived, 100. * stats.nb_survived / nb_completed_games))
    print('Games Defeated: (%d) (%.2f)' % (stats.nb_defeated, 100. * stats.nb_defeated / nb_completed_games))
    for power_name, nb_assignations in stats.nb_power_assignations.items():
        print('Played as %s: (%d) (%.2f)' % (power_name, nb_assignations, 100. * nb_assignations / nb_completed_games))
    for player_index, player_name in enumerate(players_names):
        print('Player %s played as [%s]' % (player_name,
                                            ','.join([powers[player_index] for powers in stats.assigned_powers])))
        print('Player %s ranked [%s]' % (player_name,
                                         ','.join([str(ranking[player_index]) for ranking in stats.rankings])))
    print('-' * 80)