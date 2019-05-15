import argparse
import random

from tornado import ioloop, gen

from diplomacy_research.players.player import Player
from diplomacy_research.utils.cluster import is_port_opened, kill_processes_using_port, stop_io_loop

from bench import generate_daide_game, generate_gym_game, \
                  get_client_channel, start_server, \
                  reset_unsync_wait, run_benchmark, PLAYER_FACTORIES, OPEN_PORTS, ClientWrapper
from stats.cross_convoy_stats import print_cross_convoy_stats
from stats.cross_support_stats import print_cross_support_stats
from stats.ranking_stats import print_ranking_stats

IO_LOOP = None

@gen.coroutine
def _get_benchmark_args(ai_1, ai_2, args):
    name = '1[{}]v6[{}]'.format(ai_1.name, ai_2.name)
    players = [ai_1, ai_2, ai_2, ai_2, ai_2, ai_2, ai_2]

    if args.stats == 'cross_convoy':
        stats_callback = lambda games: print_cross_convoy_stats(name, games)
    elif args.stats == 'cross_support':
        stats_callback = lambda games: print_cross_support_stats(name, games)
    elif args.stats == 'ranking':
        player_names = [player.name for player in players]
        stats_callback = lambda games: print_ranking_stats(name, games, player_names)

    if 'daide' in args.ai_1 + args.ai_2:
        players = [ClientWrapper(player, None) for player in players]

        for i, (player, _) in enumerate(players):
            if isinstance(player, Player):
                channel = yield get_client_channel()
                players[i] = ClientWrapper(player, channel)
                break

        game_generator = \
            lambda players, progress_bar: generate_daide_game(players, progress_bar, args.rules)

    else:
        game_generator = generate_gym_game

    return (game_generator, players, args.games, stats_callback)

@gen.coroutine
def main():
    """ Entry point """
    global IO_LOOP

    yield gen.sleep(2)
    try:
        ai_1 = yield PLAYER_FACTORIES[args.ai_1].make()

        if args.ai_2 == args.ai_1:
            ai_2 = ai_1
        else:
            ai_2 = yield PLAYER_FACTORIES[args.ai_2].make()

        game_generator, players, games, stats_callback = yield _get_benchmark_args(ai_1, ai_2, args)
        reset_unsync_wait()
        yield run_benchmark(game_generator, players, games, stats_callback=stats_callback)

        game_generator, players, games, stats_callback = yield _get_benchmark_args(ai_2, ai_1, args)
        reset_unsync_wait()
        yield run_benchmark(game_generator, players, games, stats_callback=stats_callback)

    except Exception as exception:
        print('Exception:', exception)
    finally:
        stop_io_loop(IO_LOOP)

if __name__ == '__main__':
    ai_names = sorted(name for name in PLAYER_FACTORIES)

    parser = argparse.ArgumentParser(description='Diplomacy ai bench',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ai-1', default='dumbbot', choices=ai_names,
                        help='ai choices: ' + ' | '.join(ai_names))
    parser.add_argument('--ai-2', default='random', choices=ai_names,
                        help='ai choices: ' + ' | '.join(ai_names))
    parser.add_argument('--games', default=10, type=int,
                        help='number of pair of games to run (default: 10)')
    parser.add_argument('--stats', default='ranking', choices=['cross_convoy',
                                                               'cross_support',
                                                               'ranking'],
                        help='the stats to get: ' + ' | '.join(['cross_convoy',
                                                                'cross_support',
                                                                'ranking']))
    parser.add_argument('--rules', default='NO_PRESS,IGNORE_ERRORS,POWER_CHOICE', help='Game rules')
    args = parser.parse_args()

    args.rules = args.rules.split(',')

    print('--ai-1=[{}] --ai-2=[{}] --games=[{}] --stats=[{}] --rules=[{}]'
          .format(args.ai_1, args.ai_2, args.games, args.stats, args.rules))

    IO_LOOP = ioloop.IOLoop.instance()
    IO_LOOP.spawn_callback(main)
    try:
        start_server(IO_LOOP)
    except KeyboardInterrupt:
        pass
    finally:
        stop_io_loop(IO_LOOP)
        for port in OPEN_PORTS:
            if is_port_opened(port):
                kill_processes_using_port(port, force=True)
