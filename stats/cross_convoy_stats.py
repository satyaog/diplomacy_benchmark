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
    nb_convoys = 0
    nb_cross_convoys = 0
    nb_effective_cross_convoys = 0
    game = Game()
    game.add_rule('IGNORE_ERRORS')
    for phase in game_json['phases']:
        if phase['name'][-1] != 'M':
            continue
        phase_data = GamePhaseData.from_dict(phase)
        game.set_phase_data(phase_data, clear_history=True)
        game.add_rule('IGNORE_ERRORS')

        # Get nb_convoys
        for _, orders in game.get_orders().items():
            for order in orders:
                order_tokens = order.split()
                if len(order_tokens) < 3:
                    continue
                if order.split()[2] == 'C':
                    nb_convoys += 1

        phase_cross_convoys = check_cross_convoy(game)
        nb_cross_convoys += len(phase_cross_convoys)

        # Determine effective cross convoys
        if len(phase_cross_convoys) > 0:
            results = {normalized_unit(unit): result for unit, result in phase_data.results.items()}
            for cross_convoy in phase_cross_convoys:
                # If the supportee's move is a failure, pass
                if len(results.get(cross_convoy.supportee_unit, ['void'])) > 0:
                    continue

                # Modify the support to hold
                modified_phase_data = GamePhaseData.from_dict(phase)
                supporter = cross_convoy.supporter
                support_order_idx = modified_phase_data.orders[supporter].index(cross_convoy.supporter_order)
                modified_phase_data.orders[supporter][support_order_idx] = to_hold(cross_convoy.supporter_order)

                # Process the phase to see if the supportee fails
                game.set_phase_data(modified_phase_data, clear_history=True)
                game.add_rule('IGNORE_ERRORS')
                modified_phase_data = game.process()
                modified_results = {normalized_unit(unit): result
                                    for unit, result in modified_phase_data.results.items()}
                if len(modified_results[cross_convoy.supportee_unit]) > 0:
                    nb_effective_cross_convoys += 1

    return nb_convoys, nb_cross_convoys, nb_effective_cross_convoys

def check_cross_convoy(game):
    """ Detect if cross power convoys exists """
    unit_power_dict = {}
    for power, units in game.get_units().items():
        for unit in units:
            unit_power_dict[normalized_unit(unit)] = power

    cross_convoys = []
    for power in game.powers:
        for order in game.get_orders(power):
            order_tokens = order.split()
            if len(order_tokens) >= 5 and order_tokens[2] == 'C':
                supportee_type = order_tokens[3]
                supportee_loc = order_tokens[4][:3]
                supportee_unit = ' '.join([supportee_type, supportee_loc])
                supportee_power = unit_power_dict.get(supportee_unit, None)

                # this is a cross support
                if supportee_power is not None and power != supportee_power:
                    cross_convoy = CrossSupport(supporter=power,
                                                supporter_order=order,
                                                supportee=supportee_power,
                                                supportee_unit=supportee_unit)
                    cross_convoys.append(cross_convoy)
    return cross_convoys

def normalized_unit(unit):
    """ Strip out the coast """
    return unit[:5]

def to_hold(order):
    """ Make any order to hold """
    unit = order.split()[:2]
    return ' '.join(unit + ['H'])

def print_cross_convoy_stats(benchmark_name, games):
    nb_games = len(games)
    nb_completed_games = len([_ for _ in games if _ is not None])

    if not nb_completed_games:
        print('-' * 80)
        print('Benchmark: %s (%d/%d games)' % (benchmark_name, nb_completed_games, nb_games))
        print()
        print('No games to get stats from.')
        print('-' * 80)
        return

    total_nb_convoys, total_nb_cross_convoys, total_nb_effective_cross_convoys = 0, 0, 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(compute_ratio, [game for game in games if game is not None])):
            nb_convoys, nb_cross_convoys, nb_effective_cross_convoys = result
            total_nb_convoys += nb_convoys
            total_nb_cross_convoys += nb_cross_convoys
            total_nb_effective_cross_convoys += nb_effective_cross_convoys

    print()
    print('-' * 80)
    print('Benchmark: %s (%d/%d games)' % (benchmark_name, nb_completed_games, nb_games))
    print()
    print('Convoys', total_nb_convoys)
    percent = 100. * total_nb_cross_convoys / total_nb_convoys if total_nb_convoys else 0.0
    print('X-Convoys', total_nb_cross_convoys, percent)
    percent = 100. * total_nb_effective_cross_convoys / total_nb_cross_convoys \
              if total_nb_cross_convoys else 0.0
    print('(Effective) X-Convoys', total_nb_effective_cross_convoys, percent)
    print('-' * 80)
