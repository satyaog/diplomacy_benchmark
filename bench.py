""" Run benchmark tests """
import logging
import os
from multiprocessing import Process
import shutil
import time
import zipfile

import gym
from tornado import gen
from tqdm import tqdm

from diplomacy import Game, Map

from diplomacy_research.models.datasets.grpc_dataset import GRPCDataset, ModelConfig
from diplomacy_research.models.gym.wrappers import LimitNumberYears, RandomizePlayers, SaveGame
from diplomacy_research.models.policy.order_based import \
    PolicyAdapter as OrderPolicyAdapter, BaseDatasetBuilder as OrderBaseDatasetBuilder
from diplomacy_research.models.policy.token_based import \
    PolicyAdapter as TokenPolicyAdapter, BaseDatasetBuilder as TokenBaseDatasetBuilder
from diplomacy_research.players import ModelBasedPlayer, RandomPlayer, RuleBasedPlayer
from diplomacy_research.players.rulesets import easy_ruleset, dumbbot_ruleset
from diplomacy_research.utils.cluster import is_port_opened, kill_processes_using_port
from diplomacy_research.utils.process import start_tf_serving, download_file, kill_subprocesses_on_exit
from diplomacy_research.settings import WORKING_DIR

LOGGER = logging.getLogger('diplomacy_research.scripts.launch_bot')
PERIOD_SECONDS = 10
MAX_SENTINEL_CHECKS = 3
MAX_TIME_BETWEEN_CHECKS = 300
SERVING_PORTS_POOL = [9500, 9501]
SERVING_PORTS = []
NOISE = 0.
TEMPERATURE = 0.1
DROPOUT_RATE = 0.
USE_BEAM = False

MODEL_AI_URL = {
    'supervised': 'http://storage.googleapis.com/ppaquette-diplomacy/files/latest_model.zip',
    'reinforcement': 'https://storage.googleapis.com/ppaquette-diplomacy/files/prev_models/20190116-model-order-v12-'
                     'epoch18-4ced2c4-rl-015251.zip'
}

NON_MODEL_AI = {
    'random': RandomPlayer(),
    'dumbbot': RuleBasedPlayer(ruleset=dumbbot_ruleset),
    'easy': RuleBasedPlayer(ruleset=easy_ruleset)
}

def launch_serving(serving_port):
    """ Launches or relaunches the TF Serving process """
    # Stop all serving child processes
    if is_port_opened(serving_port):
        kill_processes_using_port(serving_port)

    # Launching a new process
    log_file_path = os.path.join(WORKING_DIR, 'data', 'log_serving.txt')
    serving_process = Process(target=start_tf_serving,
                              args=(serving_port, WORKING_DIR),
                              kwargs={'force_cpu': True,
                                      'log_file_path': log_file_path})
    serving_process.start()
    kill_subprocesses_on_exit()

    # Waiting for port to be opened.
    for attempt_ix in range(90):
        time.sleep(10)
        if is_port_opened(serving_port):
            break
        LOGGER.info('Waiting for TF Serving to come online. - Attempt %d / %d', attempt_ix + 1, 90)
    else:
        LOGGER.error('TF Serving is not online after 15 minutes. Aborting.')
        raise RuntimeError()

    # Setting configuration
    new_config = ModelConfig(name='player', base_path='/work_dir/data/bot', version_policy=None)
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
def create_model_based_player(adapter_ctor, dataset_builder_ctor):
    """ Function to connect to TF Serving server and query orders """
    serving_port = SERVING_PORTS_POOL.pop(0)
    SERVING_PORTS.append(serving_port)

    # Start TF Serving
    launch_serving(serving_port)

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
def create_player(model_name, model_url, clean_dir=True):
    """ Function to download the latest model and create a player """
    bot_directory = os.path.join(WORKING_DIR, 'data', 'bot')
    bot_model = os.path.join(bot_directory, '%s.zip' % model_name)
    if clean_dir:
        shutil.rmtree(bot_directory, ignore_errors=True)
    os.makedirs(bot_directory, exist_ok=True)

    # Downloading model
    download_file(model_url, bot_model, force=clean_dir)

    # Unzipping file
    zip_ref = zipfile.ZipFile(bot_model, 'r')
    zip_ref.extractall(bot_directory)
    zip_ref.close()

    # Detecting model type
    if os.path.exists(os.path.join(bot_directory, 'order_based.txt')):
        LOGGER.info('Creating order-based player.')
        player = yield create_model_based_player(OrderPolicyAdapter, OrderBaseDatasetBuilder)

    elif os.path.exists(os.path.join(bot_directory, 'token_based.txt')):
        LOGGER.info('Creating token-based player.')
        player = yield create_model_based_player(TokenPolicyAdapter, TokenBaseDatasetBuilder)

    else:
        LOGGER.info('Creating rule-based player')
        player = RuleBasedPlayer(ruleset=easy_ruleset)

    # Returning
    return player

@gen.coroutine
def generate_game(players, progress_bar):
    """ Generate a game """
    env = gym.make('DiplomacyEnv-v0')
    env = LimitNumberYears(env, 35)
    env = RandomizePlayers(env, players)
    env = SaveGame(env)

    # Generating game
    env.reset()
    powers = env.get_all_powers_name()

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

def get_stats(games):
    """ Computes stats """
    nb_won, nb_most, nb_survived, nb_defeated, nb_games = 0, 0, 0, 0, len(games)
    power_assignations = {power_name: 0 for power_name in Map().powers}

    for game in games:
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
def run_benchmark(name, nb_games, player_a, player_b):
    """ Runs a benchmark (1 player A vs 6 players B)
        :param name: Name of the benchmark
        :param nb_games: The number of games to use in the benchmark
        :param player_a: The player A
        :param player_b: The player B
        :return: Nothing, but displays stats
    """
    players = [player_a, player_b, player_b, player_b, player_b, player_b, player_b]
    progress_bar = tqdm(total=nb_games)

    # Generating games
    games = yield [generate_game(players, progress_bar) for _ in range(nb_games)]

    # Computing stats
    nb_won, nb_most, nb_survived, nb_defeated, power_assignations = get_stats(games)

    # Displaying stats
    print('-' * 80)
    print('Benchmark: %s (%d games)' % (name, nb_games))
    print()
    print('Games Won: (%d) (%.2f)' % (nb_won, 100. * nb_won / nb_games))
    print('Games Most SC: (%d) (%.2f)' % (nb_most, 100. * nb_most / nb_games))
    print('Games Survived: (%d) (%.2f)' % (nb_survived, 100. * nb_survived / nb_games))
    print('Games Defeated: (%d) (%.2f)' % (nb_defeated, 100. * nb_defeated / nb_games))
    for power_name, nb_assignations in power_assignations.items():
        print('Played as %s: (%d) (%.2f)' % (power_name, nb_assignations, 100. * nb_assignations / nb_games))
    print('-' * 80)
