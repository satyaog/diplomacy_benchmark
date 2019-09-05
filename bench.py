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
from tornado.concurrent import Future
from tornado.ioloop import TimeoutError
from tqdm import tqdm
import ujson as json

from diplomacy import Game, Map, Server
from diplomacy.utils.splitter import PhaseSplitter
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

ClientWrapper = namedtuple("ClientWrapper", ["player", "channel_game"])
DaideWrapper = namedtuple("DaideWrapper", ["player", "process"])

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
_client_channel = None
_unsync_wait = 0
HOSTNAME = 'localhost'
MAIN_PORT = 9456

class DaidePlayerPlaceHolder():
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__

class ModelPlayerURLFactory():
    def __init__(self, name, policy_adapter, base_dataset_builder, url,
                 temperature=TEMPERATURE, use_beam=USE_BEAM):
        self.name = name
        self.url = url
        self.policy_adapter = policy_adapter
        self.base_dataset_builder = base_dataset_builder
        self.temperature = temperature
        self.use_beam = use_beam

    @gen.coroutine
    def make(self, clean_dir=False):
        """ Function to download the latest model and create a player """
        bot_directory = os.path.join(WORKING_DIR, 'data', 'bot_%s' % self.name)
        bot_model = os.path.join(bot_directory, '%s.zip' % self.name)
        if clean_dir:
            shutil.rmtree(bot_directory, ignore_errors=True)
        os.makedirs(bot_directory, exist_ok=True)

        model_url = self.url

        # Downloading model
        download_file(model_url, bot_model, force=clean_dir)

        # Unzipping file
        zip_ref = zipfile.ZipFile(bot_model, 'r')
        zip_ref.extractall(bot_directory)
        zip_ref.close()

        serving_port = PORTS_POOL.pop(0)
        OPEN_PORTS.append(serving_port)

        # Start TF Serving
        launch_serving(self.name, serving_port)

        # Creating player
        grpc_dataset = GRPCDataset(hostname='localhost',
                                   port=serving_port,
                                   model_name='player',
                                   signature=self.policy_adapter.get_signature(),
                                   dataset_builder=self.base_dataset_builder())
        adapter = self.policy_adapter(grpc_dataset)
        player = ModelBasedPlayer(adapter,
                                  noise=NOISE,
                                  temperature=self.temperature,
                                  dropout_rate=DROPOUT_RATE,
                                  use_beam=self.use_beam,
                                  name=self.name)

        # Validating openings
        yield player.check_openings()

        # Returning
        return player

    @gen.coroutine
    def create_model_based_player():
        """ Function to connect to TF Serving server and query orders """
        serving_port = PORTS_POOL.pop(0)
        OPEN_PORTS.append(serving_port)

        # Start TF Serving
        launch_serving(self.name, serving_port)

        # Creating player
        grpc_dataset = GRPCDataset(hostname='localhost',
                                   port=serving_port,
                                   model_name='player',
                                   signature=self.policy_adapter.get_signature(),
                                   dataset_builder=self.base_dataset_builder())
        adapter = self.policy_adapter(grpc_dataset)
        player = ModelBasedPlayer(adapter,
                                  noise=NOISE,
                                  temperature=TEMPERATURE,
                                  dropout_rate=DROPOUT_RATE,
                                  use_beam=USE_BEAM)

        # Validating openings
        yield player.check_openings()

        # Returning player
        return player

class NonModelPlayerURLFactory():
    def __init__(self, name, player_ctor):
        self.name = name
        self.player_ctor = player_ctor

    @gen.coroutine
    def make(self):
        return self.player_ctor(name=self.name)

PLAYER_FACTORIES = {
    # Model AI
    'supervised_neurips19':
        ModelPlayerURLFactory('supervised_neurips19', sl_neurips2019.PolicyAdapter,
                              sl_neurips2019.BaseDatasetBuilder,
                              'https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/neurips2019-sl_model.zip'),
    'supervised_neurips19_temp1_nobeam':
        ModelPlayerURLFactory('supervised_neurips19_temp1_nobeam', sl_neurips2019.PolicyAdapter,
                              sl_neurips2019.BaseDatasetBuilder,
                              temperature=1,
                              url='https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/neurips2019-sl_model.zip'),
    'supervised_neurips19_temp1_beam':
        ModelPlayerURLFactory('supervised_neurips19_temp1_beam', sl_neurips2019.PolicyAdapter,
                              sl_neurips2019.BaseDatasetBuilder,
                              temperature=1, use_beam=True,
                              url='https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/neurips2019-sl_model.zip'),
    'supervised_neurips19_temp0.5_beam':
        ModelPlayerURLFactory('supervised_neurips19_temp0.5_beam', sl_neurips2019.PolicyAdapter,
                              sl_neurips2019.BaseDatasetBuilder,
                              temperature=0.5, use_beam=True,
                              url='https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/neurips2019-sl_model.zip'),
    'supervised_neurips19_temp0.25_beam':
        ModelPlayerURLFactory('supervised_neurips19_temp0.25_beam', sl_neurips2019.PolicyAdapter,
                              sl_neurips2019.BaseDatasetBuilder,
                              temperature=0.25, use_beam=True,
                              url='https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/neurips2019-sl_model.zip'),
    'supervised_neurips19_temp0.1_beam':
        ModelPlayerURLFactory('supervised_neurips19_temp0.1_beam', sl_neurips2019.PolicyAdapter,
                              sl_neurips2019.BaseDatasetBuilder,
                              temperature=0.1, use_beam=True,
                              url='https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/neurips2019-sl_model.zip'),
    'reinforcement_neurips19':
        ModelPlayerURLFactory('reinforcement_neurips19', rl_neurips2019.PolicyAdapter,
                              rl_neurips2019.BaseDatasetBuilder,
                              'https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/neurips2019-rl_model.zip'),
    'supervised_v001':
        ModelPlayerURLFactory('supervised_v001', OrderPolicyAdapter, OrderBaseDatasetBuilder,
                              'https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/history/sl-model-v001.zip'),
    'supervised_v002':
        ModelPlayerURLFactory('supervised_v002', OrderPolicyAdapter, OrderBaseDatasetBuilder,
                              'https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/history/sl-model-v002.zip'),
    'supervised_v003':
        ModelPlayerURLFactory('supervised_v003', OrderPolicyAdapter, OrderBaseDatasetBuilder,
                              'https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/history/sl-model-v003.zip'),
    'supervised_v004':
        ModelPlayerURLFactory('supervised_v004', OrderPolicyAdapter, OrderBaseDatasetBuilder,
                              'https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/history/sl-model-v004.zip'),
    'supervised_v005':
        ModelPlayerURLFactory('supervised_v005', OrderPolicyAdapter, OrderBaseDatasetBuilder,
                              'https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/history/sl-model-v005.zip'),
    'supervised_v006':
        ModelPlayerURLFactory('supervised_v006', OrderPolicyAdapter, OrderBaseDatasetBuilder,
                              'https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/history/sl-model-v006.zip'),
    'supervised_v007':
        ModelPlayerURLFactory('supervised_v007', OrderPolicyAdapter, OrderBaseDatasetBuilder,
                              'https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/history/sl-model-v007.zip'),
    'supervised_v008':
        ModelPlayerURLFactory('supervised_v008', OrderPolicyAdapter, OrderBaseDatasetBuilder,
                              'https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/history/sl-model-v008.zip'),
    'without_film':
        ModelPlayerURLFactory('without_film', OrderPolicyAdapter, OrderBaseDatasetBuilder,
                              'https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/history/003_without_film.zip'),
    'gcn_8_layers':
        ModelPlayerURLFactory('gcn_8_layers', OrderPolicyAdapter, OrderBaseDatasetBuilder,
                              'https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/history/004_gcn_8_layers.zip'),
    'gcn_4_layers':
        ModelPlayerURLFactory('gcn_4_layers', OrderPolicyAdapter, OrderBaseDatasetBuilder,
                              'https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/history/005_gcn_4_layers.zip'),
    'gcn_2_layers':
        ModelPlayerURLFactory('gcn_2_layers', OrderPolicyAdapter, OrderBaseDatasetBuilder,
                              'https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/history/006_gcn_2_layers.zip'),
    'no_board':
        ModelPlayerURLFactory('no_board', OrderPolicyAdapter, OrderBaseDatasetBuilder,
                              'https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/history/007_no_board.zip'),
    'board_only':
        ModelPlayerURLFactory('board_only', OrderPolicyAdapter, OrderBaseDatasetBuilder,
                              'https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/history/008_board_only.zip'),
    'avg_embedding':
        ModelPlayerURLFactory('avg_embedding', OrderPolicyAdapter, OrderBaseDatasetBuilder,
                              'https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/history/009_avg_embedding.zip'),
    'transformer_order_based':
        ModelPlayerURLFactory('transformer_order_based', OrderPolicyAdapter, OrderBaseDatasetBuilder,
                              'https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/history/010_transformer_order_based.zip'),
    'lstm_token_based':
        ModelPlayerURLFactory('lstm_token_based', TokenPolicyAdapter, TokenBaseDatasetBuilder,
                              'https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/history/011_lstm_token_based.zip'),
    'transformer_token_based':
        ModelPlayerURLFactory('transformer_token_based', TokenPolicyAdapter, TokenBaseDatasetBuilder,
                              'https://f002.backblazeb2.com/file/ppaquette-public'
                              '/benchmarks/history/012_transformer_token_based.zip'),
    # Non model AI
    'random': NonModelPlayerURLFactory('random', RandomPlayer),
    'dumbbot': NonModelPlayerURLFactory('dumbbot', lambda name: RuleBasedPlayer(ruleset=dumbbot_ruleset,
                                                                                name=name)),
    'easy': NonModelPlayerURLFactory('easy', lambda name: RuleBasedPlayer(ruleset=easy_ruleset,
                                                                          name=name)),
    'daide_albert_v6.0.1': NonModelPlayerURLFactory('daide_albert_v6.0.1', DaidePlayerPlaceHolder)
}

def get_server():
    global _server
    return _server

def start_server(io_loop):
    global _server

    if _server is not None:
        io_loop.stop()

    _server = Server()
    _server.start(port=MAIN_PORT, io_loop=io_loop)

@gen.coroutine
def get_client_channel():
    global _server
    global _client_channel

    client_channel = None

    if _client_channel is None:
        username = 'user'
        password = 'password'
        connection = yield connect(HOSTNAME, MAIN_PORT)
        _client_channel = yield connection.authenticate(
            username, password, create_user=not _server.users.has_user(username, password))

    return _client_channel

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
    for attempt_idx in range(30):
        time.sleep(10)
        if is_port_opened(serving_port):
            break
        LOGGER.info('Waiting for TF Serving to come online. - Attempt %d / %d', attempt_idx + 1, 30)
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

def compute_ranking(power_names, nb_centers, elimination_orders):
    ranking_data = list(zip(power_names, nb_centers, elimination_orders))
    # ording based on nb_centers
    ranking_data.sort(key=lambda element: -element[1])
    ranking = [0] * 7
    for i, (power_name, nb_centers, _) in enumerate(ranking_data):
        rank = max(ranking) + 1
        if i > 0:
            previous_power_name, previous_nb_centers, _ = ranking_data[i - 1]
            if nb_centers == previous_nb_centers:
                rank = ranking[power_names.index(previous_power_name)]

        ranking[power_names.index(power_name)] = rank

    for i, (power_name, nb_centers, elimination_order) in enumerate(ranking_data):
        if i > 0:
            previous_power_name, previous_nb_centers, _ = ranking_data[i - 1]
            if elimination_order:
                ranking[power_names.index(power_name)] += \
                    max(elimination_orders) - elimination_order

    return ranking

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

    elimination_orders = {"_0": 0}

    yield gen.sleep(get_unsync_wait())

    while not env.is_done:
        orders = yield [player.get_orders(env.game, power_name) for (player, power_name) in zip(players, powers)]
        for power_name, power_orders in zip(powers, orders):
            env.step((power_name, power_orders))
        env.process()

        elimination_order = max(elimination_orders.values()) + 1
        for power_name in powers:
            if power_name not in elimination_orders and not env.game.get_power(power_name).units:
                elimination_orders[power_name] = elimination_order

    elimination_orders = [elimination_orders.get(power_name, 0) for power_name in powers]
    nb_centers = [len(env.game.get_power(power_name).centers) for power_name in powers]

    # Returning game

    game = env.get_saved_game()

    game['assigned_powers'] = powers
    game['ranking'] = compute_ranking(powers, nb_centers, elimination_orders)
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
    power_names = [power_names[idx] for idx in players_ordering]
    clients = {power_name: player for power_name, player in zip(power_names, players)}
    nb_daide_players = len([_ for _, (player, _) in clients.items() if isinstance(player, DaidePlayerPlaceHolder)])
    nb_regular_players = min(1, len(power_names) - nb_daide_players)

    server_game = ServerGame(n_controls=nb_daide_players + nb_regular_players,
                             rules=daide_rules)
    server_game.server = _server

    _server.add_new_game(server_game)

    reg_power_name, reg_client = None, None

    for power_name, (player, channel) in clients.items():
        if channel:
            game = yield channel.join_game(game_id=server_game.game_id, power_name=power_name)
            reg_power_name, reg_client = power_name, ClientWrapper(player, game)
            clients[power_name] = reg_client
        elif isinstance(player, Player):
            server_game.get_power(power_name).set_controlled(player.name)

    if nb_daide_players:
        server_port = PORTS_POOL.pop(0)
        OPEN_PORTS.append(server_port)
        _server.start_new_daide_server(server_game.game_id, port=server_port)
        yield gen.sleep(1)

        for power_name, (player, _) in clients.items():
            if isinstance(player, DaidePlayerPlaceHolder):
                process = launch_daide_client(server_port)
                clients[power_name] = DaideWrapper(player, process)

    for attempt_idx in range(30):
        if server_game.count_controlled_powers() == len(power_names):
            break
        yield gen.sleep(10)
        LOGGER.info('Waiting for DAIDE to connect. - Attempt %d / %d', attempt_idx + 1, 30)
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

    elimination_orders = {"_0": 0}

    yield gen.sleep(get_unsync_wait())

    try:
        watched_game = reg_client.channel_game if reg_client else server_game
        phase = PhaseSplitter(watched_game.get_current_phase())
        while watched_game.status != strings.COMPLETED and phase.year < max_year:
            print('\n=== NEW PHASE ===\n')
            print(watched_game.get_current_phase())

            if reg_client:
                yield reg_client.channel_game.wait()

            players_orders = yield [player.get_orders(server_game, power_name)
                                    for power_name, (player, _) in clients.items() if power_name in local_powers]

            for power_name, orders in zip(local_powers, players_orders):
                if phase.type == 'R':
                    orders = [order.replace(' - ', ' R ') for order in orders]
                orders = [order for order in orders if order != 'WAIVE']
                server_game.set_orders(power_name, orders, expand=False)

            while reg_client and not server_game.get_power(reg_power_name).order_is_set:
                orders = yield reg_client.player.get_orders(server_game, reg_power_name)
                print('Sending orders')
                yield reg_client.channel_game.set_orders(orders=orders)

            print('All orders sent')

            if reg_client:
                yield reg_client.channel_game.no_wait()

            for attempt_idx in range(120):
                if phase.input_str != watched_game.get_current_phase() or \
                        server_game.status != strings.ACTIVE:
                    break
                if (attempt_idx + 1) % 12 == 0 and phase.input_str != server_game.get_current_phase():
                    # Watched game is unsynched
                    watched_game = server_game
                    break
                LOGGER.info('Waiting for the phase to be processed. - Attempt %d / %d', attempt_idx + 1, 120)
                yield gen.sleep(2.5)
            else:
                LOGGER.error('Phase is taking too long to process. Aborting.')
                raise RuntimeError()

            elimination_order = max(elimination_orders.values()) + 1
            for power_name in power_names:
                if power_name not in elimination_orders and not server_game.get_power(power_name).units:
                    elimination_orders[power_name] = elimination_order
                    if power_name == reg_power_name:
                        watched_game = server_game

            if server_game.status != strings.ACTIVE:
                break

            phase = PhaseSplitter(watched_game.get_current_phase())

    except TimeoutError as timeout:
        print('Timeout: ', timeout)
    except Exception as exception:
        print('Exception: ', exception)
    finally:
        _server.stop_daide_server(server_game.game_id)
        yield gen.sleep(1)
        for power_name, (player, _) in clients.items():
            if isinstance(player, DaidePlayerPlaceHolder):
                process = clients[power_name].process
                process.kill()
        if reg_client:
            reg_client.channel_game.leave()

    game = None
    saved_game = to_saved_game_format(server_game)

    if server_game.status == strings.COMPLETED or PhaseSplitter(server_game.get_current_phase()).year >= max_year:
        elimination_orders = [elimination_orders.get(power_name, 0) for power_name in power_names]
        nb_centers = [len(server_game.get_power(power_name).centers) for power_name in power_names]

        game = saved_game
        game['assigned_powers'] = power_names
        game['ranking'] = compute_ranking(power_names, nb_centers, elimination_orders)

    with open('game_{}.json'.format(saved_game['id']), 'w') as file:
        json.dump(saved_game, file)

    progress_bar.update()

    return game

@gen.coroutine
def run_benchmark(game_generator, players, nb_games, progress_bar=None, stats_callback=None):
    """ Runs a benchmark
    """

    if progress_bar is None:
        progress_bar = tqdm(total=nb_games)

    # Generating games
    games = yield [game_generator(players, progress_bar) for _ in range(nb_games)]

    if stats_callback:
        # Wait until all the server printing is done
        yield gen.sleep(5)
        # new line for the stats printing
        print()
        stats_callback(games)
