""" Run benchmark tests """
from collections import namedtuple
import logging
import os
from multiprocessing import Process
from random import randint, shuffle
import shutil
import subprocess
import time
import zipfile

import gym
from tornado import gen
from tornado.ioloop import TimeoutError
from tqdm import tqdm
import ujson as json

from diplomacy import Game, Map, Server
from diplomacy.utils.subject_split import PhaseSplit
from diplomacy.server.server_game import ServerGame
from diplomacy.client.connection import connect
from diplomacy.utils import strings
from diplomacy.utils.export import to_saved_game_format

from diplomacy_research.models.datasets.grpc_dataset import GRPCDataset, ModelConfig
from diplomacy_research.models.gym.wrappers import LimitNumberYears, RandomizePlayers, SaveGame
from diplomacy_research.models.policy.order_based import \
    PolicyAdapter as OrderPolicyAdapter, BaseDatasetBuilder as OrderBaseDatasetBuilder
from diplomacy_research.models.policy.token_based import \
    PolicyAdapter as TokenPolicyAdapter, BaseDatasetBuilder as TokenBaseDatasetBuilder
from diplomacy_research.players.benchmarks import rl_neurips2019, sl_neurips2019
from diplomacy_research.players.player import Player
from diplomacy_research.players import ModelBasedPlayer, RandomPlayer, RuleBasedPlayer
from diplomacy_research.players.rulesets import easy_ruleset, dumbbot_ruleset
from diplomacy_research.utils.cluster import is_port_opened, kill_processes_using_port
from diplomacy_research.utils.process import start_tf_serving, download_file, kill_subprocesses_on_exit
from diplomacy_research.settings import WORKING_DIR

ModelURLBuilder = namedtuple("ModelURLBuilder", ["url", "builder"])
ModelBuilder = namedtuple("ModelBuilder", ["PolicyAdapter", "BaseDatasetBuilder"])
ClientPlayer = namedtuple("ClientPlayer", ["player", "game"])
DaidePlayer = namedtuple("DaidePlayer", ["player", "process"])

LOGGER = logging.getLogger('diplomacy_research.scripts.launch_bot')
PERIOD_SECONDS = 10
MAX_SENTINEL_CHECKS = 3
MAX_TIME_BETWEEN_CHECKS = 300
PORTS_POOL = [9500+i for i in range(100)]
OPEN_PORTS = []
NOISE = 0.
TEMPERATURE = 0.1
DROPOUT_RATE = 0.
USE_BEAM = False

_server = None
_unsync_wait = 0
HOSTNAME = 'localhost'
MAIN_PORT = 9456

MODEL_AI_URL_BUILDER = {
    'reinforcement': ModelURLBuilder('https://storage.googleapis.com/ppaquette-diplomacy/'
                                     'files/prev_models/20190116-model-order-v12-epoch18-'
                                     '4ced2c4-rl-015251.zip',
                                     ModelBuilder(OrderPolicyAdapter, OrderBaseDatasetBuilder)),
    'supervised': ModelURLBuilder('http://storage.googleapis.com/ppaquette-diplomacy/'
                                  'files/latest_model.zip',
                                  ModelBuilder(OrderPolicyAdapter, OrderBaseDatasetBuilder)),
    'reinforcement_neurips2019': ModelURLBuilder('http://www-ens.iro.umontreal.ca/~paquphil/'
                                                 'benchmarks/neurips2019-rl_model.zip',
                                                 rl_neurips2019),
    'supervised_neurips2019': ModelURLBuilder('http://www-ens.iro.umontreal.ca/~paquphil/'
                                              'benchmarks/neurips2019-sl_model.zip',
                                              sl_neurips2019)
}

NON_MODEL_AI = {
    'random': RandomPlayer(),
    'dumbbot': RuleBasedPlayer(ruleset=dumbbot_ruleset),
    'easy': RuleBasedPlayer(ruleset=easy_ruleset)
}

def start_server(io_loop):
    global _server

    if _server is not None:
        _server.stop()

    _server = Server()
    _server.start(port=MAIN_PORT, io_loop=io_loop)

def get_unsync_wait():
    global _unsync_wait
    unsync_wait = _unsync_wait
    _unsync_wait += 0.5
    return unsync_wait

def reset_unsync_wait():
    global _unsync_wait
    _unsync_wait = 0

def launch_daide_client(port):
    process = subprocess.Popen(["singularity", "run", "albert_dumbbot-1.1.img", "albert", "127.0.0.1", str(port)],
                               cwd=os.path.join(WORKING_DIR, 'data', 'bot'))
    return process

def launch_serving(model_name, serving_port):
    """ Launches or relaunches the TF Serving process """
    # Stop all serving child processes
    if is_port_opened(serving_port):
        kill_processes_using_port(serving_port)

    # Launching a new process
    log_file_path = os.path.join(WORKING_DIR, 'data', 'log_serving_%d.txt' % serving_port)
    serving_process = Process(target=start_tf_serving,
                              args=(serving_port, WORKING_DIR),
                              kwargs={'force_cpu': True,
                                      'log_file_path': log_file_path})
    serving_process.start()
    kill_subprocesses_on_exit()

    # Waiting for port to be opened.
    for attempt_ix in range(30):
        time.sleep(10)
        if is_port_opened(serving_port):
            break
        LOGGER.info('Waiting for TF Serving to come online. - Attempt %d / %d', attempt_ix + 1, 30)
    else:
        LOGGER.error('TF Serving is not online after 5 minutes. Aborting.')
        raise RuntimeError()

    # Setting configuration
    new_config = ModelConfig(name='player', base_path='/work_dir/data/bot_%s' % model_name, version_policy=None)
    for _ in range(30):
        if GRPCDataset.set_config('localhost', serving_port, new_config):
            LOGGER.info('Configuration set successfully.')
            break
        time.sleep(5.)
    else:
        LOGGER.error('Unable to set the configuration file.')

@gen.coroutine
def check_serving(player):
    """ Makes sure the current serving process is still active, otherwise restarts it.
        :param player: A player object to query the server
    """
    game = Game()

    # Trying to check orders
    for _ in range(MAX_SENTINEL_CHECKS):
        orders = yield player.get_orders(game, 'FRANCE')
        if orders:
            return

    # Could not get orders x times in a row, restarting process
    LOGGER.warning('Could not retrieve orders from the serving process after %d attempts.', MAX_SENTINEL_CHECKS)
    LOGGER.warning('Restarting TF serving server.')
    launch_serving()

@gen.coroutine
def create_model_based_player(model_name, adapter_ctor, dataset_builder_ctor):
    """ Function to connect to TF Serving server and query orders """
    serving_port = PORTS_POOL.pop(0)
    OPEN_PORTS.append(serving_port)

    # Start TF Serving
    launch_serving(model_name, serving_port)

    # Creating player
    grpc_dataset = GRPCDataset(hostname='localhost',
                               port=serving_port,
                               model_name='player',
                               signature=adapter_ctor.get_signature(),
                               dataset_builder=dataset_builder_ctor())
    adapter = adapter_ctor(grpc_dataset)
    player = ModelBasedPlayer(adapter,
                              noise=NOISE,
                              temperature=TEMPERATURE,
                              dropout_rate=DROPOUT_RATE,
                              use_beam=USE_BEAM)

    # Validating openings
    yield player.check_openings()

    # Returning player
    return player

@gen.coroutine
def create_player(model_name, model_url_builder, clean_dir=True):
    """ Function to download the latest model and create a player """
    bot_directory = os.path.join(WORKING_DIR, 'data', 'bot_%s' % model_name)
    bot_model = os.path.join(bot_directory, '%s.zip' % model_name)
    if clean_dir:
        shutil.rmtree(bot_directory, ignore_errors=True)
    os.makedirs(bot_directory, exist_ok=True)

    model_url = model_url_builder.url

    # Downloading model
    download_file(model_url, bot_model, force=clean_dir)

    # Unzipping file
    zip_ref = zipfile.ZipFile(bot_model, 'r')
    zip_ref.extractall(bot_directory)
    zip_ref.close()

    # Detecting model type
    if os.path.exists(os.path.join(bot_directory, 'order_based.txt')):
        policy_adapter = model_url_builder.builder.PolicyAdapter
        dataset_builder = model_url_builder.builder.BaseDatasetBuilder
        LOGGER.info('Creating order-based player.')
        player = yield create_model_based_player(model_name, policy_adapter, dataset_builder)

    elif os.path.exists(os.path.join(bot_directory, 'token_based.txt')):
        LOGGER.info('Creating token-based player.')
        player = yield create_model_based_player(model_name, TokenPolicyAdapter, TokenBaseDatasetBuilder)

    else:
        LOGGER.info('Creating rule-based player')
        player = RuleBasedPlayer(ruleset=easy_ruleset)

    # Returning
    return player

@gen.coroutine
def generate_gym_game(players, progress_bar):
    """ Generate a game """
    env = gym.make('DiplomacyEnv-v0')
    env = LimitNumberYears(env, 35)
    env = RandomizePlayers(env, players)
    env = SaveGame(env)

    # Generating game
    env.reset()
    powers = env.get_all_powers_name()

    yield gen.sleep(get_unsync_wait())

    while not env.is_done:
        orders = yield [player.get_orders(env.game, power_name) for (player, power_name) in zip(players, powers)]
        for power_name, power_orders in zip(powers, orders):
            env.step((power_name, power_orders))
        env.process()

    # Returning game
    game = env.get_saved_game()
    game['assigned_powers'] = powers
    progress_bar.update()
    return game

@gen.coroutine
def generate_daide_game(players, progress_bar, daide_rules):
    """ Generate a game """
    global _server

    max_number_of_year = 35
    max_year = Map().first_year + max_number_of_year

    players_ordering = list(range(len(players)))
    shuffle(players_ordering)
    power_names = Map().powers
    clients = {power_names[idx]: player for idx, player in zip(players_ordering, players)}
    nb_daide_players = len([_ for _, (player, _) in clients.items() if not isinstance(player, Player)])
    nb_regular_players = 1

    server_game = ServerGame(n_controls=nb_daide_players + nb_regular_players,
                             rules=daide_rules)
    server_game.server = _server

    _server.add_new_game(server_game)

    reg_power_name, reg_client = None, None

    for power_name, (player, channel) in clients.items():
        if channel:
            game = yield channel.join_game(game_id=server_game.game_id, power_name=power_name)
            reg_power_name, reg_client = power_name, ClientPlayer(player, game)
            clients[power_name] = reg_client
        elif isinstance(player, Player):
            server_game.get_power(power_name).set_controlled(type(player).__name__)

    if nb_daide_players:
        server_port = PORTS_POOL.pop(0)
        OPEN_PORTS.append(server_port)
        _server.start_new_daide_server(server_game.game_id, port=server_port)
        yield gen.sleep(1)

        for power_name, (player, _) in clients.items():
            if not isinstance(player, Player):
                process = launch_daide_client(server_port)
                clients[power_name] = DaidePlayer(player, process)

    for attempt_ix in range(30):
        yield gen.sleep(10)
        if server_game.count_controlled_powers() == len(power_names):
            break
        LOGGER.info('Waiting for DAIDE to connect. - Attempt %d / %d', attempt_ix + 1, 30)
    else:
        LOGGER.error('DAIDE is not online after 5 minutes. Aborting.')
        raise RuntimeError()

    for power_name, (player, game) in clients.items():
        if not game and isinstance(player, Player):
            server_game.get_power(power_name).set_controlled(strings.DUMMY)

    if server_game.game_can_start():
        _server.start_game(server_game)

    local_powers = [power_name for power_name, (player, game) in clients.items()
                    if not game and isinstance(player, Player)]

    yield gen.sleep(get_unsync_wait())

    try:
        phase = PhaseSplit.split(reg_client.game.get_current_phase())
        watched_game = reg_client.game
        while watched_game.status != strings.COMPLETED and phase.year < max_year:
            print('\n=== NEW PHASE ===\n')
            print(watched_game.get_current_phase())

            yield reg_client.game.wait()

            players_orders = yield [player.get_orders(server_game, power_name)
                                    for power_name, (player, _) in clients.items() if power_name in local_powers]

            for power_name, orders in zip(local_powers, players_orders):
                if phase.type == 'R':
                    orders = [order.replace(' - ', ' R ') for order in orders]
                orders = [order for order in orders if order != 'WAIVE']
                server_game.set_orders(power_name, orders, expand=False)

            while not server_game.get_power(reg_power_name).order_is_set:
                orders = yield reg_client.player.get_orders(server_game, reg_power_name)
                print('Sending orders')
                yield reg_client.game.set_orders(orders=orders)

            print('All orders sent')

            yield reg_client.game.no_wait()

            while phase.in_str == watched_game.get_current_phase():
                print('Waiting for the phase to be processed')
                yield gen.sleep(10)

            if not reg_client.game.power.units:
                watched_game = server_game

            if watched_game.get_current_phase().lower() == strings.COMPLETED:
                break

            phase = PhaseSplit.split(watched_game.get_current_phase())

    except TimeoutError as timeout:
        print('Timeout: ', timeout)
    except Exception as exception:
        print('Exception: ', exception)
    finally:
        _server.stop_daide_server(server_game.game_id)
        yield gen.sleep(1)
        for power_name, (player, _) in clients.items():
            if not isinstance(player, Player):
                process = clients[power_name].process
                process.kill()
        if reg_client:
            reg_client.game.leave()

    game = None
    if server_game.status == strings.COMPLETED or PhaseSplit.split(server_game.get_current_phase()).year >= max_year:
        game = to_saved_game_format(server_game)
        game['assigned_powers'] = list(clients.keys())
        with open('game_{}.json'.format(game['id']), 'w') as file:
            json.dump(game, file)

    progress_bar.update()

    return game

def get_stats(games):
    """ Computes stats """
    nb_won, nb_most, nb_survived, nb_defeated = 0, 0, 0, 0
    power_assignations = {power_name: 0 for power_name in Map().powers}

    for game in games:
        if not game:
            continue

        assigned_powers = game['assigned_powers']
        nb_centers = {power_name: len(game['phases'][-1]['state']['centers'][power_name])
                      for power_name in assigned_powers}
        if nb_centers[assigned_powers[0]] >= 18:
            nb_won += 1
        elif nb_centers[assigned_powers[0]] == max(nb_centers.values()):
            nb_most += 1
        elif nb_centers[assigned_powers[0]] > 0:
            nb_survived += 1
        else:
            nb_defeated += 1
        power_assignations[assigned_powers[0]] += 1

    return nb_won, nb_most, nb_survived, nb_defeated, power_assignations

@gen.coroutine
def run_benchmark(generate_game, name, nb_games, player_a, player_b):
    """ Runs a benchmark (1 player A vs 6 players B)
        :param name: Name of the benchmark
        :param nb_games: The number of games to use in the benchmark
        :param player_a: The player A
        :param player_b: The player B
        :return: Nothing, but displays stats
    """
    global _server

    players = [player_a, player_b, player_b, player_b, player_b, player_b, player_b]
    progress_bar = tqdm(total=nb_games)

    if _server and _server.backend is not None:
        players = [ClientPlayer(player, None) for player in players]

        for i in range(len(players)):
            player, _ = players[i]
            if isinstance(player, Player):
                username = 'user'
                password = 'password'
                connection = yield connect(HOSTNAME, MAIN_PORT)
                channel = yield connection.authenticate(username, password,
                                                        create_user=not _server.users.has_user(username, password))
                players[i] = ClientPlayer(player, channel)
                break

    # Generating games
    reset_unsync_wait()
    games = yield [generate_game(players, progress_bar) for _ in range(nb_games)]

    # Computing stats
    nb_won, nb_most, nb_survived, nb_defeated, power_assignations = get_stats(games)

    nb_completed_games = len([_ for _ in games if _ is not None])

    yield gen.sleep(5)
    # Displaying stats
    print('\n'+'-' * 80)
    print('Benchmark: %s (%d/%d games)' % (name, nb_completed_games, nb_games))
    print()
    print('Games Won: (%d) (%.2f)' % (nb_won, 100. * nb_won / nb_completed_games))
    print('Games Most SC: (%d) (%.2f)' % (nb_most, 100. * nb_most / nb_completed_games))
    print('Games Survived: (%d) (%.2f)' % (nb_survived, 100. * nb_survived / nb_completed_games))
    print('Games Defeated: (%d) (%.2f)' % (nb_defeated, 100. * nb_defeated / nb_completed_games))
    for power_name, nb_assignations in power_assignations.items():
        print('Played as %s: (%d) (%.2f)' % (power_name, nb_assignations, 100. * nb_assignations / nb_completed_games))
    print('-' * 80+'\n')
