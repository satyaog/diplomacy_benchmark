import re
import argparse

def to_csv(filename, player, opponent):
    search_replaces = [(r"(../results/)?bench_[\w_\.\d]+: *\d*",
                        r""),
                       (r"^(\w+): ([\d\w_\[\]]*) \(\d\d?/?\d?\d? games\)|"  # Benchmark: ... (3/5 games)                        # \1 \2
                        r"^([ \w]+): \((\d+)\) \([\d\.]+\)|"                # Games Won: (0) (0.00)                             # \3 \4
                        r"^Player [\w\._]+ [\w ]+\[[A-Z]+[,A-Z]*\]|"        # Player daide_albert_v6.0.1 played as [ENGLAND]
                        r"^Player [\w\._]+ [\w ]+\[\d+[,\d]*\]|"            # Player daide_albert_v6.0.1 ranked [1]
                        r"^-{80}\n?",
                        r"\1\3\n\2\4"),
                       (r"^\n",
                        r""),
                       (r"(Benchmark)\n(1\[{player}\]v6\[{opponent}\])\n"   # \1  \2
                        r"([\w ]+)\n(\d+)\n"    # Games Won\n0              # \3  \4
                        r"([\w ]+)\n(\d+)\n"    # Games Most\n0             # \5  \6
                        r"([\w ]+)\n(\d+)\n"    # Games Survived\n1         # \7  \8
                        r"([\w ]+)\n(\d+)\n"    # Games Defeated\n2         # \9  \10
                        r"([\w ]+)\n(\d+)\n"    # Played as AUSTRIA\n0      # \11 \12 
                        r"([\w ]+)\n(\d+)\n"    # Played as ENGLAND\n0      # \13 \14
                        r"([\w ]+)\n(\d+)\n"    # Played as FRANCE\n1       # \15 \16
                        r"([\w ]+)\n(\d+)\n"    # Played as GERMANY\n2      # \17 \18
                        r"([\w ]+)\n(\d+)\n"    # Played as ITALY\n0        # \19 \20
                        r"([\w ]+)\n(\d+)\n"    # Played as RUSSIA\n0       # \21 \22
                        r"([\w ]+)\n(\d+)\n"    # Played as TURKEY\n0       # \23 \24
                        r"(Benchmark)\n(1\[{opponent}\]v6\[{player}\])\n"   # \25 \26
                        r"([\w ]+)\n(\d+)\n"    # Games Won\n0              # \27 \28
                        r"([\w ]+)\n(\d+)\n"    # Games Most\n0             # \29 \30
                        r"([\w ]+)\n(\d+)\n"    # Games Survived\n1         # \31 \32
                        r"([\w ]+)\n(\d+)\n"    # Games Defeated\n2         # \33 \34
                        r"([\w ]+)\n(\d+)\n"    # Played as AUSTRIA\n0      # \35 \36
                        r"([\w ]+)\n(\d+)\n"    # Played as ENGLAND\n0      # \37 \38
                        r"([\w ]+)\n(\d+)\n"    # Played as FRANCE\n1       # \39 \40
                        r"([\w ]+)\n(\d+)\n"    # Played as GERMANY\n2      # \41 \42
                        r"([\w ]+)\n(\d+)\n"    # Played as ITALY\n0        # \43 \44
                        r"([\w ]+)\n(\d+)\n"    # Played as RUSSIA\n0       # \45 \46
                        r"([\w ]+)\n(\d+)"      # Played as TURKEY\n0       # \47 \48
                        .format(player=player, opponent=opponent),
                        r"\2,,,,,,,,,,,"                            r"\26\n"
                        r"\3,\5,\7,\9,\11,\13,\15,\17,\19,\21,\23," r"\27,\29,\31,\33,\35,\37,\39,\41,\43,\45,\47\n"
                        r"\4,\6,\8,\10,\12,\14,\16,\18,\20,\22,\24,"r"\28,\30,\32,\34,\36,\38,\40,\42,\44,\46,\48"),
                       (r"\n1\[{player}\]v6\[{opponent}\],,,,,,,,,,,1\[{opponent}\]v6\[{player}\]\n"
                        r"Games Won,Games Most SC,Games Survived,Games Defeated,Played as AUSTRIA,Played as ENGLAND,Played as FRANCE,Played as GERMANY,Played as ITALY,Played as RUSSIA,Played as TURKEY,Games Won,Games Most SC,Games Survived,Games Defeated,Played as AUSTRIA,Played as ENGLAND,Played as FRANCE,Played as GERMANY,Played as ITALY,Played as RUSSIA,Played as TURKEY\n"
                        .format(player=player, opponent=opponent),
                        r"\n")]

    with open(filename, "r") as file:
        content = file.read()

    for search_replace in search_replaces:
        content = re.sub(search_replace[0], search_replace[1], content, flags=re.MULTILINE)

    return content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert winning stats to csv",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("results", metavar="RESULTS_FILE",
                        help="path to the results file. Data file is expected to "
                             "have been extracted with\n"
                             "> egrep -E -e \"^-{80}|^Benchmark|^Games |^Played as |^Player \" LOG_FILES*.stats* > RESULTS_FILE")
    parser.add_argument("--ai-1", help="The name of the AI that was benched in "
                                       "1v6 and 6v1 games")
    parser.add_argument("--ai-2", help="The name of the AI that was the opponent "
                                       "in 1v6 and 6v1 games")
    parser.add_argument("-o", "--output-csv", default=None,
                        help="The csv to save")
    args = parser.parse_args()

    csv_content = to_csv(args.results, args.ai_1, args.ai_2)
    if args.output_csv:
        with open(args.output_csv, "w") as csv_file:
            csv_file.write(csv_content)
    else:
        print(csv_content)
