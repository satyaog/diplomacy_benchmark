import argparse
import glob
import os
import random

from tornado import ioloop, gen
from tqdm import tqdm
import ujson as json

from diplomacy_research.players.player import Player
from diplomacy_research.utils.cluster import is_port_opened, kill_processes_using_port, stop_io_loop

from bench import generate_daide_game, generate_gym_game, \
                  get_client_channel, start_server, \
                  reset_unsync_wait, run_benchmark, PLAYER_FACTORIES, OPEN_PORTS, ClientWrapper
from stats.save_games import save_games
from stats.ranking_stats import print_ranking_stats

def callback_array(games, callbacks):
    for cb in callbacks:
        cb(games)

@gen.coroutine
def _get_benchmark_args(players_choices, args):
    name = 'TrueSkill'
    players = random.sample(list(players_choices.values()), 7)
    is_daide_game = 'daide' in '_'.join([player.name for player in players])

    if is_daide_game:
        players = [ClientWrapper(player, None) for player in players]
        name += '_w_DAIDE'

        for i, (player, _) in enumerate(players):
            if isinstance(player, Player):
                channel = yield get_client_channel()
                players[i] = ClientWrapper(player, channel)
                break

        game_generator = \
            lambda players, progress_bar: generate_daide_game(players, progress_bar, args.rules)

    else:
        game_generator = generate_gym_game

    callbacks = []
    for stats_name in args.stats:
        if stats_name == 'save_games':
            stats_callback = lambda games: save_games(args.save_dir, games)
        elif stats_name == 'ranking':
            stats_callback = lambda games: print_ranking_stats(name, games)
        else:
            continue

        callbacks.append(stats_callback)

    callback = lambda games: callback_array(games, callbacks)

    return (is_daide_game, game_generator, players, callback)

IO_LOOP = None

@gen.coroutine
def main():
    """ Entry point """
    global IO_LOOP

    if args.seed is not None:
        random.seed(args.seed)

    yield gen.sleep(2)
    try:
        players_choices = yield {name: factory.make() for name, factory in PLAYER_FACTORIES.items()
                                                      if not args.exclude_daide or 'daide' not in name}

        gym_benchmark_kwargs = []
        daide_benchmark_kwargs = []

        while len(gym_benchmark_kwargs) + len(daide_benchmark_kwargs) < args.games:
            is_daide_game, game_generator, players, callback = yield _get_benchmark_args(players_choices, args)
            benchmark_kwargs = {
                'game_generator': game_generator,
                'players': players,
                'callback': callback
            }

            if is_daide_game:
                daide_benchmark_kwargs.append(benchmark_kwargs)

            else:
                gym_benchmark_kwargs.append(benchmark_kwargs)

        if gym_benchmark_kwargs:
            reset_unsync_wait()
            nb_games = len(gym_benchmark_kwargs)
            progress_bar = tqdm(total=nb_games)
            yield [run_benchmark(kwargs['game_generator'], kwargs['players'], nb_games=1,
                                 progress_bar=progress_bar, stats_callback=kwargs['callback'])
                   for kwargs in gym_benchmark_kwargs]

        if daide_benchmark_kwargs:
            reset_unsync_wait()
            nb_games = len(daide_benchmark_kwargs)
            progress_bar = tqdm(total=nb_games)
            yield [run_benchmark(kwargs['game_generator'], kwargs['players'], nb_games=1,
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
    parser.add_argument('--stats', default='ranking',
                        help='a comma separated list of stats to get: ' +
                             ' | '.join(['ranking']))
    parser.add_argument('--save-dir', default=None,
                        help='the directory to save games')
    parser.add_argument('--existing-games-dir', default=None,
                        help='the directory containing the games to load instead '
                             'of running new games')
    parser.add_argument('--rules', default='NO_PRESS,IGNORE_ERRORS,POWER_CHOICE',
                        help='Game rules')
    parser.add_argument('--exclude-daide', default=False, action='store_true',
                        help='Exclude DAIDE models')
    parser.add_argument('--seed', default=None, type=int, help='Seed to use')
    args = parser.parse_args()

    args.stats = [stat for stat in args.stats.split(',') if stat]
    args.rules = [rule for rule in args.rules.split(',') if rule]

    if args.save_dir:
        args.stats = ['save_games'] + args.stats

    if args.existing_games_dir:
        games = []
        glob_pattern = os.path.join(args.existing_games_dir, "game_*.json")
        filenames = glob.glob(glob_pattern)
        for filename in filenames:
            with open(filename, "r") as file:
                content = file.read()
            games.append(json.loads(content.rstrip('\n')))

        args.games = len(games)
        try: args.stats.remove('save_games')
        except ValueError: pass
        args.save_dir = None
        args.exclude_daide = None
        args.rules = None
        args.seed = None
        print('--games=[{}] --stats=[{}] --save-dir=[{}] --existing-games-dir=[{}] '
              '--rules=[{}] --exclude-daide=[{}] --seed=[{}]'
              .format(args.games, args.stats, args.save_dir, args.existing_games_dir,
                      args.rules, args.exclude_daide, args.seed))

        callbacks = []
        name = os.path.abspath(glob_pattern)
        for stats_name in args.stats:
            if stats_name == 'ranking':
                stats_callback = lambda games: print_ranking_stats(name, games)
            else:
                continue

            callbacks.append(stats_callback)

        for game in games:
            callback_array([game], callbacks)

    else:
        print('--games=[{}] --stats=[{}] --save-dir=[{}] --existing-games-dir=[{}] '
              '--rules=[{}] --exclude-daide=[{}] --seed=[{}]'
              .format(args.games, args.stats, args.save_dir, args.existing_games_dir,
                      args.rules, args.exclude_daide, args.seed))

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
