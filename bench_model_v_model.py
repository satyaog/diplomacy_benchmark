import argparse

from tornado import gen

from diplomacy_research.utils.cluster import is_port_opened, kill_processes_using_port, start_io_loop, stop_io_loop

from bench import create_player, run_benchmark, MODEL_AI_URL, SERVING_PORTS

model_ai_names = sorted(name for name in MODEL_AI_URL
                 if name.islower() and not name.startswith("__")
                 and isinstance(MODEL_AI_URL[name], str))

parser = argparse.ArgumentParser(description='Diplomacy model vs model bench')
parser.add_argument('--model-ai', default='supervised',
                    choices=model_ai_names,
                    help='model ai: ' +
                         ' | '.join(model_ai_names) +
                         ' (default: supervised)')
parser.add_argument('--other-model-ai', default='reinforcement',
                    choices=model_ai_names,
                    help='model ai: ' +
                         ' | '.join(model_ai_names) +
                         ' (default: supervised)')
parser.add_argument('--games', default=10, type=int,
                    help='number of pair of games to run (default: 10)')
args = parser.parse_args()

@gen.coroutine
def main():
    """ Entry point """
    print('--games=[{}]'.format(args.games))
    try:
        player = yield create_player(args.model_ai, MODEL_AI_URL[args.model_ai], clean_dir=False)
        opponent = None
        if args.other_model_ai == args.model_ai:
            opponent = player
        else:
            opponent = yield create_player(args.other_model_ai, MODEL_AI_URL[args.other_model_ai], clean_dir=False)
        yield run_benchmark('1[{}]v6[{}]'.format(args.model_ai, args.other_model_ai), args.games, player, opponent)
        yield run_benchmark('1[{}]v6[{}]'.format(args.other_model_ai, args.model_ai), args.games, opponent, player)
    except Exception as exception:
        print('Exception:', exception)
    finally:
        stop_io_loop()
        for serving_port in SERVING_PORTS:
            if is_port_opened(serving_port):
                kill_processes_using_port(serving_port, force=True)

if __name__ == '__main__':
    start_io_loop(main)
