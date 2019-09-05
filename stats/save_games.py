import ujson as json

def save_games(games):
    for game in games:
        if game is not None:
            with open('game_{}.json'.format(game['id']), 'w') as file:
                json.dump(game, file)
