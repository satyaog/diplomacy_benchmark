import argparse

from tornado import ioloop, gen

from diplomacy_research.players.player import Player
from diplomacy_research.utils.cluster import is_port_opened, kill_processes_using_port, start_io_loop, stop_io_loop

from bench import create_player, generate_daide_game, start_server, run_benchmark, MODEL_AI_URL_BUILDER, OPEN_PORTS

model_ai_names = sorted(name for name in MODEL_AI_URL_BUILDER
                 if name.islower() and not name.startswith("__")
                 and isinstance(MODEL_AI_URL_BUILDER[name], str))

parser = argparse.ArgumentParser(description='Diplomacy DAIDE vs Rules bench')
parser.add_argument('--model-ai', default='supervised',
                    choices=model_ai_names,
                    help='model ai: ' +
                         ' | '.join(model_ai_names) +
                         ' (default: supervised)')
parser.add_argument('--games', default=10, type=int,
                    help='number of pair of games to run (default: 10)')
args = parser.parse_args()

IO_LOOP = None

@gen.coroutine
def main():
    """ Entry point """
    yield gen.sleep(2)
    print('--model-ai=[{}] --games=[{}]'.format(args.model_ai, args.games))
    try:
        player = yield create_player(args.model_ai, MODEL_AI_URL_BUILDER[args.model_ai], clean_dir=False)
        opponent = None
        yield run_benchmark(generate_daide_game, '1[{}]v6[{}]'.format(args.model_ai, 'DAIDE'), args.games,
                            player, opponent)
        yield run_benchmark(generate_daide_game, '1[{}]v6[{}]'.format('DAIDE', args.model_ai), args.games,
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
