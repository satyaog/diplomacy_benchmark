import argparse
from collections import namedtuple
import re
import io

from trueskill import TrueSkill

RankingData = namedtuple("RankingData", ["player_names", "rankings"])
PlayerGameRank = namedtuple("PlayerGameRank", ["name", "power", "rank"])

def extract_ranking_data(filename):
    player_names = {}
    rankings = []
    search_replaces = [(r"(../results/)?bench_[\w_\.\d]+: *\d*",
                        r""),
                       (r"^\w+: [\d\w_\[\]]* \(\d\d?/?\d?\d? games\)|"      # Benchmark: ... (3/5 games)
                        r"^[ \w]+: \(\d+\) \([\d\.]+\)|"                    # Games Won: (0) (0.00)
                        r"^Player ([\w\._]+) [\w ]+\[([A-Z]+)[,A-Z]*\]|"    # Player daide_albert_v6.0.1 played as [ENGLAND]    # \1 \2
                        r"^Player ([\w\._]+) [\w ]+\[(\d+)[,\d]*\]|"        # Player daide_albert_v6.0.1 ranked [1]             # \3 \4
                        r"^-{80}\n?|"                                       # ---...
                        r"^.+\n?",
                        r"\1\n\2\4"),
                       (r"^\n",
                        r""),
                       (r"([\w\._]+)\n([A-Z]+)\n(\d+)\n"    # daide_albert_v6.0.1\nENGLAND\n1   # \1  \2  \3
                        r"([\w\._]+)\n([A-Z]+)\n(\d+)\n"    # supervised_v005\nAUSTRIA\n6       # \4  \5  \6
                        r"([\w\._]+)\n([A-Z]+)\n(\d+)\n"    # easy\nITALY\n7                    # \7  \8  \9
                        r"([\w\._]+)\n([A-Z]+)\n(\d+)\n"    # supervised_neurips19\nRUSSIA\n3   # \10 \11 \12
                        r"([\w\._]+)\n([A-Z]+)\n(\d+)\n"    # supervised_v003\nTURKEY\n4        # \13 \14 \15
                        r"([\w\._]+)\n([A-Z]+)\n(\d+)\n"    # supervised_v007\nFRANCE\n2        # \16 \17 \18
                        r"([\w\._]+)\n([A-Z]+)\n(\d+)\n",   # supervised_v006\nGERMANY\n5       # \19 \20 \21
                        r"\1,\4,\7,\10,\13,\16,\19\n"
                        r"\2,\5,\8,\11,\14,\17,\20\n"
                        r"\3,\6,\9,\12,\15,\18,\21\n")]

    with open(filename, "r") as file:
        results_content = file.read()

    for search_replace in search_replaces:
        results_content = re.sub(search_replace[0], search_replace[1], results_content, flags=re.MULTILINE)

    buffer = io.StringIO(results_content)
    string = buffer.readline()
    while string:
        game_player_names = string.rstrip().split(',')
        string = buffer.readline()
        powers = string.rstrip().split(',')
        string = buffer.readline()
        ranks = string.rstrip().split(',')
        string = buffer.readline()

        player_names.update({name: '' for name in game_player_names})
        game_ranking = {name: PlayerGameRank(name, power, int(rank))
                        for name, power, rank in zip(game_player_names, powers, ranks)}
        rankings.append(game_ranking)

    player_names = list(player_names.keys())
    player_names.sort()

    ranking_data = RankingData(player_names, rankings)

    return ranking_data

def compute_trueskill(player_names, rankings):
    env = TrueSkill()
    ratings = {name: env.create_rating() for name in player_names}

    for game_ranking in rankings:
        game_player_names, ranks = zip(*[(player_name, rank) for player_name, (_, _, rank) in game_ranking.items()])
        rating_groups = env.rate([(ratings[player_name],) for player_name in game_player_names], ranks)
        for player_name, (rating,) in zip(game_player_names, rating_groups):
            ratings[player_name] = rating

    return ratings

def ratings_and_rankings_to_csv(player_ratings, rankings):
    player_names = player_ratings.keys()

    buffer = [",".join(["Player Name", "Mu", "Sigma"] * len(player_ratings))]
    line_buffer = []
    for player_name, rating in player_ratings.items():
        line_buffer.append(player_name)
        line_buffer.append(str(rating.mu))
        line_buffer.append(str(rating.sigma))
    buffer.append(",".join(line_buffer))
    buffer.append(",".join(["Power", "Rank", ""] * len(player_ratings)))

    for game_ranking in rankings:
        line_buffer = []
        for player_name in player_names:
            player_game_rank = game_ranking.get(player_name, None)
            if player_game_rank:
                line_buffer.append(player_game_rank.power)
                line_buffer.append(str(player_game_rank.rank))
                line_buffer.append('')
            else:
                line_buffer.append('')
                line_buffer.append('')
                line_buffer.append('')
        buffer.append(",".join(line_buffer))

    return "\n".join(buffer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute TrueSkill from games bench results",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("results", metavar="RESULTS_FILE",
                        help="path to the results file. Data file is expected to "
                             "have been extracted with\n"
                             "> egrep -E -e \"^-{80}|^Benchmark|^Games |^Played as |^Player \" LOG_FILES*.stats* > RESULTS_FILE")
    parser.add_argument("-o", "--output-csv", default=None,
                        help="Save a csv with the TrueSkill ratings")
    args = parser.parse_args()

    player_names, rankings = extract_ranking_data(args.results)
    ratings = compute_trueskill(player_names, rankings)
    csv_content = ratings_and_rankings_to_csv(ratings, rankings)
    if args.output_csv:
        with open(args.output_csv, "w") as csv_file:
            csv_file.write(csv_content)
    else:
        print(csv_content)
