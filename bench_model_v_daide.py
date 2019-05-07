import argparse

from tornado import ioloop, gen

from diplomacy_research.players.player import Player
from diplomacy_research.utils.cluster import is_port_opened, kill_processes_using_port, start_io_loop, stop_io_loop

from bench import create_player, generate_daide_game, start_server, run_benchmark, MODEL_AI_URL, OPEN_PORTS

model_ai_names = sorted(name for name in MODEL_AI_URL
                 if name.islower() and not name.startswith("__")
                 and isinstance(MODEL_AI_URL[name], str))

parser = argparse.ArgumentParser(description='Diplomacy mixed bench',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--games', default=10, type=int,
                    help='number of pair of games to run')
parser.add_argument('--daide-games', default=10, type=int,
                    help='number of pair of DAIDE games to run')
parser.add_argument('--rules', default=['SOLITAIRE', 'NO_PRESS', 'IGNORE_ERRORS', 'POWER_CHOICE'],
                    nargs='*', help='Game rules')
args = parser.parse_args()

IO_LOOP = None

@gen.coroutine
def main():
    """ Entry point """
    yield gen.sleep(2)
    print('--model-ai=[{}] --games=[{}] --no-press=[{}]'
          .format(args.model_ai, args.games, args.no_press))
    try:
        if args.no_press:
            daide_rules = ['NO_PRESS', 'IGNORE_ERRORS', 'POWER_CHOICE']
        else:
            daide_rules = ['IGNORE_ERRORS', 'POWER_CHOICE']

        def generate_game(players, progress_bar):
            return generate_daide_game(players, progress_bar, daide_rules)

        player = yield create_player(args.model_ai, MODEL_AI_URL[args.model_ai], clean_dir=False)
        opponent = None
        yield run_benchmark(generate_game, '1[{}]v6[{}]'.format(args.model_ai, 'DAIDE'), args.games,
                            player, opponent)
        yield run_benchmark(generate_game, '1[{}]v6[{}]'.format('DAIDE', args.model_ai), args.games,
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
