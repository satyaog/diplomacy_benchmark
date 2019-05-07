import argparse

from tornado import gen

from diplomacy_research.utils.cluster import is_port_opened, kill_processes_using_port, start_io_loop, stop_io_loop

from research.bench import create_player, generate_gym_game, run_benchmark, MODEL_AI_URL, OPEN_PORTS

parser = argparse.ArgumentParser(description='Diplomacy model vs model bench')
parser.add_argument('--games', default=10, type=int,
                    help='number of pair of games to run (default: 10)')
args = parser.parse_args()

@gen.coroutine
def main():
    """ Entry point """
    print('--games=[{}]'.format(args.games))
    try:
        player = yield create_player('supervised', MODEL_AI_URL['supervised'], clean_dir=False)
        opponent = yield create_player('reinforcement', MODEL_AI_URL['reinforcement'], clean_dir=False)
        yield run_benchmark(generate_gym_game, '1[{}]v6[{}]'.format('supervised', 'reinforcement'), args.games, player, opponent)
        yield run_benchmark(generate_gym_game, '1[{}]v6[{}]'.format('reinforcement', 'supervised'), args.games, opponent, player)
    except Exception as exception:
        print('Exception:', exception)
    finally:
        stop_io_loop()
        for port in OPEN_PORTS:
            if is_port_opened(port):
                kill_processes_using_port(port, force=True)

if __name__ == '__main__':
    start_io_loop(main)
