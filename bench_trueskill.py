import argparse
import random

from tornado import ioloop, gen
from tqdm import tqdm

from diplomacy_research.players.player import Player
from diplomacy_research.utils.cluster import is_port_opened, kill_processes_using_port, stop_io_loop

from stats.ranking_stats import print_ranking_stats
from bench import generate_daide_game, generate_gym_game, \
                  get_client_channel, start_server, \
                  reset_unsync_wait, run_benchmark, PLAYER_FACTORIES, OPEN_PORTS, ClientWrapper

IO_LOOP = None

@gen.coroutine
def main():
    """ Entry point """
    global IO_LOOP

    if args.seed is not None:
        random.seed(args.seed)

    yield gen.sleep(2)
    try:
        players_choices = yield {name: factory.make() for name, factory in PLAYER_FACTORIES.items()}

        gym_benchmark_kwargs = []
        daide_benchmark_kwargs = []

        while len(gym_benchmark_kwargs) + len(daide_benchmark_kwargs) < args.games:
            players = random.sample(list(players_choices.values()), 7)
            player_names = [player.name for player in players]

            is_daide_game = False
            benchmark_kwargs = {
                'generate_game': None,
                'name': '',
                'players': None,
                'callback': None
            }

            for player in players:
                if 'daide' in player.name:
                    is_daide_game = True
                    break

            if is_daide_game:
                players = [ClientWrapper(player, None) for player in players]

                for i, (player, _) in enumerate(players):
                    if isinstance(player, Player):
                        channel = yield get_client_channel()
                        players[i] = ClientWrapper(player, channel)
                        break

                generate_game = \
                    lambda players, progress_bar: generate_daide_game(players, progress_bar, args.rules)
                name = 'DAIDE_games'
                callback = lambda games: print_ranking_stats(name, games, player_names)
                benchmark_kwargs['generate_game'] = generate_game
                benchmark_kwargs['players'] = players
                benchmark_kwargs['callback'] = callback
                daide_benchmark_kwargs.append(benchmark_kwargs)

            else:
                name = ''
                callback = lambda games: print_ranking_stats(name, games, player_names)
                benchmark_kwargs['generate_game'] = generate_gym_game
                benchmark_kwargs['players'] = players
                benchmark_kwargs['callback'] = callback
                gym_benchmark_kwargs.append(benchmark_kwargs)

        if gym_benchmark_kwargs:
            reset_unsync_wait()
            nb_games = len(gym_benchmark_kwargs)
            progress_bar = tqdm(total=nb_games)
            yield [run_benchmark(kwargs['generate_game'], kwargs['players'], nb_games=1,
                                 progress_bar=progress_bar, stats_callback=kwargs['callback'])
                   for kwargs in gym_benchmark_kwargs]

        if daide_benchmark_kwargs:
            reset_unsync_wait()
            nb_games = len(daide_benchmark_kwargs)
            progress_bar = tqdm(total=nb_games)
            yield [run_benchmark(kwargs['generate_game'], kwargs['players'], nb_games=1,
                                 progress_bar=progress_bar, stats_callback=kwargs['callback'])
                   for kwargs in daide_benchmark_kwargs]

    except Exception as exception:
        print('Exception:', exception)
    finally:
        stop_io_loop(IO_LOOP)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diplomacy mixed bench',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--games', default=10, type=int,
                        help='number of games to run')
    parser.add_argument('--rules', default='NO_PRESS,IGNORE_ERRORS,POWER_CHOICE', help='Game rules')
    parser.add_argument('--seed', default=None, type=int, help='Seed to use')
    args = parser.parse_args()

    args.rules = args.rules.split(',')

    print('--games=[{}] --rules=[{}] --seed=[{}]'.format(args.games, args.rules, args.seed))

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
