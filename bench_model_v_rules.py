import argparse

from tornado import gen

from diplomacy_research.players.player import Player
from diplomacy_research.utils.cluster import is_port_opened, kill_processes_using_port, start_io_loop, stop_io_loop

from bench import create_player, generate_gym_game, run_benchmark, MODEL_AI_URL_BUILDER, NON_MODEL_AI, OPEN_PORTS

model_ai_names = sorted(name for name in MODEL_AI_URL_BUILDER
                 if name.islower() and not name.startswith("__")
                 and isinstance(MODEL_AI_URL_BUILDER[name], str))
non_model_ai_names = sorted(name for name in NON_MODEL_AI
                     if name.islower() and not name.startswith("__")
                     and isinstance(NON_MODEL_AI[name], Player))

parser = argparse.ArgumentParser(description='Diplomacy Model vs Rules bench')
parser.add_argument('--model-ai', default='supervised',
                    choices=model_ai_names,
                    help='model ai: ' +
                         ' | '.join(model_ai_names) +
                         ' (default: supervised)')
parser.add_argument('--non-model-ai', default='random',
                    choices=non_model_ai_names,
                    help='non model ai: ' +
                         ' | '.join(non_model_ai_names) +
                         ' (default: random)')
parser.add_argument('--games', default=10, type=int,
                    help='number of pair of games to run (default: 10)')
args = parser.parse_args()

@gen.coroutine
def main():
    """ Entry point """
    print('--model-ai=[{}] --non-model-ai=[{}] --games=[{}]'.format(args.model_ai, args.non_model_ai, args.games))
    try:
        player = yield create_player(args.model_ai, MODEL_AI_URL_BUILDER[args.model_ai], clean_dir=False)
        opponent = NON_MODEL_AI[args.non_model_ai]
        yield run_benchmark(generate_gym_game, '1[{}]v6[{}]'.format(args.model_ai, args.non_model_ai), args.games, player, opponent)
        yield run_benchmark(generate_gym_game, '1[{}]v6[{}]'.format(args.non_model_ai, args.model_ai), args.games, opponent, player)
    except Exception as exception:
        print('Exception:', exception)
    finally:
        stop_io_loop()
        for port in OPEN_PORTS:
            if is_port_opened(port):
                kill_processes_using_port(port, force=True)

if __name__ == '__main__':
    start_io_loop(main)
