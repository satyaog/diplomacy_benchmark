"""
Try to compute the success cross support ratio in a game
"""
import concurrent.futures
from tqdm import tqdm
from diplomacy import Game
from diplomacy.utils.game_phase_data import GamePhaseData
import ujson as json
from collections import namedtuple

class CrossSupport(namedtuple('CrossSupport', ('supporter',
                                               'supporter_order',
                                               'supportee',
                                               'supportee_unit'))):
    pass

def compute_ratio(game_json):
    if game_json['map'] != 'standard':
        return 0, 0, 0
    nb_supports = 0
    nb_cross_supports = 0
    nb_effective_cross_supports = 0
    game = Game()
    game.add_rule('IGNORE_ERRORS')
    for phase in game_json['phases']:
        if phase['name'][-1] != 'M':
            continue
        phase_data = GamePhaseData.from_dict(phase)
        game.set_phase_data(phase_data, clear_history=True)
        game.add_rule('IGNORE_ERRORS')

        # Get nb_supports
        for _, orders in game.get_orders().items():
            for order in orders:
                order_tokens = order.split()
                if len(order_tokens) < 3:
                    continue
                if order.split()[2] == 'S':
                    nb_supports += 1

        phase_cross_supports = check_cross_support(game)
        nb_cross_supports += len(phase_cross_supports)

        # Determine effective cross supports
        if len(phase_cross_supports) > 0:
            results = {normalized_unit(unit): result for unit, result in phase_data.results.items()}
            for cross_support in phase_cross_supports:
                # If the supportee's move is a failure, pass
                if len(results.get(cross_support.supportee_unit, ['void'])) > 0:
                    continue

                # Modify the support to hold
                modified_phase_data = GamePhaseData.from_dict(phase)
                supporter = cross_support.supporter
                support_order_idx = modified_phase_data.orders[supporter].index(cross_support.supporter_order)
                modified_phase_data.orders[supporter][support_order_idx] = to_hold(cross_support.supporter_order)

                # Process the phase to see if the supportee fails
                game.set_phase_data(modified_phase_data, clear_history=True)
                game.add_rule('IGNORE_ERRORS')
                modified_phase_data = game.process()
                modified_results = {normalized_unit(unit): result
                                    for unit, result in modified_phase_data.results.items()}
                if len(modified_results[cross_support.supportee_unit]) > 0:
                    nb_effective_cross_supports += 1

    return nb_supports, nb_cross_supports, nb_effective_cross_supports

def check_cross_support(game):
    """ Detect if cross power support exists """
    unit_power_dict = {}
    for power, units in game.get_units().items():
        for unit in units:
            unit_power_dict[normalized_unit(unit)] = power

    cross_supports = []
    for power in game.powers:
        for order in game.get_orders(power):
            order_tokens = order.split()
            if len(order_tokens) >= 5 and order_tokens[2] == 'S':
                supportee_type = order_tokens[3]
                supportee_loc = order_tokens[4][:3]
                supportee_unit = ' '.join([supportee_type, supportee_loc])
                supportee_power = unit_power_dict.get(supportee_unit, None)

                # this is a cross support
                if supportee_power is not None and power != supportee_power:
                    cross_support = CrossSupport(supporter=power,
                                                 supporter_order=order,
                                                 supportee=supportee_power,
                                                 supportee_unit=supportee_unit)
                    cross_supports.append(cross_support)
    return cross_supports

def normalized_unit(unit):
    """ Strip out the coast """
    return unit[:5]

def to_hold(order):
    """ Make any order to hold """
    unit = order.split()[:2]
    return ' '.join(unit + ['H'])

def print_cross_support_stats(benchmark_name, games):
    nb_games = len(games)
    nb_completed_games = len([_ for _ in games if _ is not None])

    if not nb_completed_games:
        print('-' * 80)
        print('Benchmark: %s (%d/%d games)' % (benchmark_name, nb_completed_games, nb_games))
        print()
        print('No games to get stats from.')
        print('-' * 80)
        return

    total_nb_supports, total_nb_cross_supports, total_nb_effective_cross_supports = 0, 0, 0

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(compute_ratio, [game for game in games if game is not None])):
            nb_supports, nb_cross_supports, nb_effective_cross_supports = result
            total_nb_supports += nb_supports
            total_nb_cross_supports += nb_cross_supports
            total_nb_effective_cross_supports += nb_effective_cross_supports

    print()
    print('-' * 80)
    print('Benchmark: %s (%d/%d games)' % (benchmark_name, nb_completed_games, nb_games))
    print()
    print('Supports', total_nb_supports)
    percent = 100. * total_nb_cross_supports / total_nb_supports if total_nb_supports else 0.0
    print('X-Supports', total_nb_cross_supports, percent)
    percent = 100. * total_nb_effective_cross_supports / total_nb_cross_supports \
              if total_nb_cross_supports else 0.0
    print('(Effective) X-Supports', total_nb_effective_cross_supports, percent)
    print('-' * 80)