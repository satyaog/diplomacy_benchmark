import os
import ujson as json

def save_games(save_dir, games):
    try: os.makedirs(save_dir)
    except FileExistsError: pass
    for game in games:
        if game is not None:
            filename = 'game_{}.json'.format(game['id'])
            filename = os.path.join(save_dir, filename)
            with open(filename, 'w') as file:
                json.dump(game, file)
                print('Game saved at [{}]'.format(os.path.abspath(filename)))
