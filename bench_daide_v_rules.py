import argparse

from tornado import ioloop, gen

from diplomacy_research.players.player import Player
from diplomacy_research.utils.cluster import is_port_opened, kill_processes_using_port, stop_io_loop

from bench import generate_daide_game, start_server, run_benchmark, NON_MODEL_AI, OPEN_PORTS

non_model_ai_names = sorted(name for name in NON_MODEL_AI
                     if name.islower() and not name.startswith("__")
                     and isinstance(NON_MODEL_AI[name], Player))

parser = argparse.ArgumentParser(description='Diplomacy DAIDE vs Rules bench')
parser.add_argument('--non-model-ai', default='random',
                    choices=non_model_ai_names,
                    help='non model ai: ' +
                         ' | '.join(non_model_ai_names) +
                         ' (default: random)')
parser.add_argument('--games', default=10, type=int,
                    help='number of pair of games to run (default: 10)')
parser.add_argument('--no-press', default=False, action='store_true',
                    help='Disable press messages in game. (default: False)')
args = parser.parse_args()

IO_LOOP = None

@gen.coroutine
def main():
    """ Entry point """
    yield gen.sleep(2)
    print('--non-model-ai=[{}] --games=[{}] --no-press=[{}]'.format(args.non_model_ai, args.games, args.no_press))
    try:
        if args.no_press:
            daide_rules = ['NO_PRESS', 'IGNORE_ERRORS', 'POWER_CHOICE']
        else:
            daide_rules = ['IGNORE_ERRORS', 'POWER_CHOICE']

        def generate_game(players, progress_bar):
            return generate_daide_game(players, progress_bar, daide_rules)

        player = None
        opponent = NON_MODEL_AI[args.non_model_ai]
        yield run_benchmark(generate_game, '1[{}]v6[{}]'.format('DAIDE', args.non_model_ai), args.games,
                            player, opponent)
        yield run_benchmark(generate_game, '1[{}]v6[{}]'.format(args.non_model_ai, 'DAIDE'), args.games,
                            opponent, player)
    except Exception as exception:
        print('Exception:', exception)
    finally:
        stop_io_loop(IO_LOOP)

if __name__ == '__main__':
    IO_LOOP = ioloop.IOLoop.instance()
    IO_LOOP.spawn_callback(main)
    try:
        start_server(IO_LOOP)
    except KeyboardInterrupt:
        IO_LOOP.stop()
    finally:
        for port in OPEN_PORTS:
            if is_port_opened(port):
                kill_processes_using_port(port, force=True)
